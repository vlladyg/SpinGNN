from typing import List, Optional, Tuple
from math import sqrt

import torch
from torch import fx

from e3nn import o3
from e3nn.util.jit import compile
from e3nn.util import prod
from e3nn.o3 import Instruction

from opt_einsum_fx import jitable, optimize_einsums_full


def ETN_third_order_step_forward(
    base_in1: o3.Irreps,
    mul_in1: int,
    base_in2: o3.Irreps,
    mul_in2: int,
    base_out: o3.Irreps,
    mul_out: int,
    num_paths: int,
) -> Optional[fx.GraphModule]:
    """Returns next feature vector"""
    
    #w3j = (
    #    w3j.to_dense()
    #    .reshape(((num_paths,) if num_paths > 1 else tuple()) + kij_shape)
    #    .contiguous()
    #)

    # Generate the mixer
    w3j_shape = (num_paths,) + (base_out.dim, base_in1.dim, base_in2.dim)
    C_shape = (num_paths,) + (mul_in1, mul_in2, mul_out)

    # generate actual code
    graph_out = fx.Graph()
    tracer = fx.proxy.GraphAppendingTracer(graph_out)

    def Proxy(n):
        return fx.Proxy(n, tracer=tracer)

    # = Function definitions =
    u_in = Proxy(graph_out.placeholder("u_in", torch.Tensor))
    F = Proxy(graph_out.placeholder("F", torch.Tensor))

    w3j = Proxy(graph_out.placeholder("w3j", torch.Tensor))
    w3j = w3j.reshape(w3j_shape)
    
    C = Proxy(graph_out.placeholder("C", torch.Tensor))
    C = C.reshape(C_shape)
    
    
    # convert to strided
    u_in = u_in.reshape(-1, base_in1.dim, C_shape[1])
    F = F.reshape(-1, base_in2.dim, C_shape[2])

    # do the einsum
    
    einstr = f"puvw,ziu,zjv,pkij->zkw"
    u_out = torch.einsum(einstr, C, u_in, F, w3j)
    
    graph_out.output(u_out.node)

    # check graphs
    graph_out.lint()

    # Make GraphModules
    # By putting the constants in a Module rather than a dict,
    # we force FX to copy them as buffers instead of as attributes.
    #
    # FX seems to have resolved this issue for dicts in 1.9, but we support all the way back to 1.8.0.
    constants_root = torch.nn.Module()
    #constants_root.register_buffer("_big_w3j", w3j)
    
    graphmod_out = fx.GraphModule(constants_root, graph_out, class_name="etn_step_forward")

    if True:  # optimize_einsums
        # Note that for our einsums, we can optimize _once_ for _any_ batch dimension
        # and still get the right path for _all_ batch dimensions.
        # This is because our einsums are essentially of the form:
        #    zuvw,ijk,zuvij->zwk    OR     uvw,ijk,zuvij->zwk
        # In the first case, all but one operands have the batch dimension
        #    => The first contraction gains the batch dimension
        #    => All following contractions have batch dimension
        #    => All possible contraction paths have cost that scales linearly in batch size
        #    => The optimal path is the same for all batch sizes
        # For the second case, this logic follows as long as the first contraction is not between the first two operands. Since those two operands do not share any indexes, contracting them first is a rare pathological case. See
        # https://github.com/dgasmith/opt_einsum/issues/158
        # for more details.
        #
        # TODO: consider the impact maximum intermediate result size on this logic
        #         \- this is the `memory_limit` option in opt_einsum
        # TODO: allow user to choose opt_einsum parameters?
        #
        # We use float32 and zeros to save memory and time, since opt_einsum_fx looks only at traced shapes, not values or dtypes.
        batchdim = 4
        example_inputs = (
            torch.zeros((batchdim, base_in1.dim * mul_in1)),
            torch.zeros((batchdim, base_in2.dim * mul_in2)),
            torch.zeros(w3j_shape),
            torch.zeros(
                1,
                prod(C_shape),
            ),
        )
        graphmod_out = jitable(optimize_einsums_full(graphmod_out, example_inputs))

    graphmod_out.C_shape = C_shape
    graphmod_out._dim_in1 = base_in1.dim
    graphmod_out._dim_in2 = base_in2.dim
    graphmod_out._dim_out = base_out.dim
    graphmod_out._mul_out = mul_out
    graphmod_out.C_numel = abs(prod(C_shape))

    return graphmod_out


def Contracter_ETN_ALS(
    base_in1: o3.Irreps,
    mul_in1: int,
    base_in2: o3.Irreps,
    mul_in2: int,
    base_out: o3.Irreps,
    mul_out: int,
    num_paths: int,
):
    
    mod = ETN_third_order_step_forward(
        base_in1 = base_in1,
        mul_in1 = mul_in1,
        base_in2 = base_in2,
        mul_in2 = mul_in2,
        base_out = base_out,
        mul_out = mul_out,
        num_paths = num_paths,
    )

    mod = compile(mod)
    return mod