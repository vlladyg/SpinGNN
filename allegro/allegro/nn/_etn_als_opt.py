from typing import Optional, List
import math
import functools

import torch
from torch import nn
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.utils.tp_utils import tp_path_exists

from ._fc import ScalarMLPFunction
from .. import _keys

from ._strided import Contracter_ETN_ALS
from .cutoffs import cosine_cutoff, polynomial_cutoff
from e3nn.o3 import wigner_3j
from torch.nn import Parameter, ParameterList


# Triangular ineguality for path existance
def tri_ineq(l1, l2, l3):
    return max([l1, l2, l3]) <= min([l1 + l2, l2 + l3, l1 + l3])


@compile_mode("script")
class ETN_ALS_Module_opt(nn.Module, GraphModuleMixin):
    def __init__(self,
                 d: int,
                 N_rank_ett: List[int], 
                 irreps_in=None,
                 out_field: str = AtomicDataDict.PER_ATOM_ENERGY_KEY):
        
        super().__init__()
        self.out_field = out_field
        
        
        self.d = d
        self.Nc = irreps_in[_keys.NODE_FEATURES_F][0][0]
        self.register_buffer("N_rank_ett", torch.as_tensor(N_rank_ett, dtype=torch.long))
        
        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                _keys.NODE_FEATURES_F
            ],
            irreps_out={_keys.NODE_FEATURES_ETN: o3.Irreps(
                    [(self.Nc, ir) for _, ir in irreps_in[_keys.NODE_FEATURES_F] ]),
                        out_field: o3.Irreps([(1, (0, 1))])}
        )
        
        
        # Parameters of the network
        
        # tensors for atomic features encoding
        lmax = irreps_in[_keys.EDGE_FEATURES_F].lmax # maximum spherical harmonic
        self.lmax = lmax
        
        # Second order cores(first and last)
        core2_1 = Parameter(torch.empty(lmax+1, 1, self.Nc, N_rank_ett[0]).normal_())
        core2_d = Parameter(torch.empty(lmax+1, N_rank_ett[-1], self.Nc, 1).normal_())


        instrusctions_1 = [(0, l, l) for l in range(lmax + 1)]
        instrusctions_d = [(l, l, 0) for l in range(lmax + 1)]
        
        
        # Third order cores
        # Assume irreps does not change 
        base_in1 = o3.Irreps([el[1] for el in irreps_in[_keys.EDGE_FEATURES_F]])
        base_in2 = o3.Irreps([el[1] for el in irreps_in[_keys.EDGE_FEATURES_F]])
        base_out = o3.Irreps([el[1] for el in irreps_in[_keys.EDGE_FEATURES_F]])
        

        # Building instructions
        instructions: List[Tuple[int, int, int]] = []
        tmp_i_out: int = 0
        for i_out, (_, ir_out) in enumerate(base_out):
            for i_1, (_, ir_in1) in enumerate(base_in1):
                for i_2, (_, ir_in2) in enumerate(base_in2):
                    if ir_out in ir_in1 * ir_in2:
                        instructions.append((i_1, i_2, i_out))
        
                        tmp_i_out += 1

                        
        self.instructions = instructions
        self.instructions_list = [instrusctions_1] + [instructions for _ in range(d - 2)] + [instrusctions_d]
        
        # building large w3j
        w3j_values = []
        w3j_index = []
        for i_in1, i_in2, i_out in instructions:
            mul_ir_in1 = base_in1[i_in1]
            mul_ir_in2 = base_in2[i_in2]
            mul_ir_out = base_out[i_out]
    
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert (
                tri_ineq(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
            )
    
            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                raise ValueError
    
            this_w3j = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)
            this_w3j_index = this_w3j.nonzero()
            w3j_values.append(
                this_w3j[this_w3j_index[:, 0], this_w3j_index[:, 1], this_w3j_index[:, 2]]
            )
    
            
            this_w3j_index[:, 0] += base_in1[: i_in1].dim
            this_w3j_index[:, 1] += base_in2[: i_in2].dim
            this_w3j_index[:, 2] += base_out[: i_out].dim
            # Now need to flatten the index to be for [pk][ij]
            w3j_index.append(
                torch.cat(
                    (   this_w3j_index[:, 2].unsqueeze(-1),
                        this_w3j_index[:, 0].unsqueeze(-1) * base_in2.dim
                        + this_w3j_index[:, 1].unsqueeze(-1),
                    ),
                    dim=1,
                )
            )
    
        num_paths: int = len(instructions)
    
        w3j = torch.sparse_coo_tensor(
            indices=torch.cat(w3j_index, dim=0).t(),
            values=torch.cat(w3j_values, dim=0),
            size=(
                num_paths * base_out.dim,
                base_in1.dim * base_in2.dim,
            ),
        ).coalesce()
        
        # in dense, must shape it for einsum:
        kij_shape = (
            base_out.dim,
            base_in1.dim,
            base_in2.dim,
        )
        
        # save to buffer in sparce mode + shape
        self.register_buffer("w3j", w3j)
        self.w3j_shape = (num_paths, ) + kij_shape
        
        # third order free parameters
        self.cores = ParameterList([core2_1] + [Parameter(torch.empty(num_paths, N_rank_ett[r], self.Nc, N_rank_ett[r+1]).normal_()) for r in range(d - 2)] + [core2_d]) 
        
        #self.reset_parameters()


        #print(w3j.device)
        # Register layers
        self.tps = [Contracter_ETN_ALS(base_in1, 
                                   N_rank_ett[r], 
                                   base_in2, 
                                   self.Nc, 
                                   base_out, 
                                   N_rank_ett[r+1], 
                                   num_paths) for r in range(self.d - 2)]

        
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        # Input features
        F = data[_keys.NODE_FEATURES_F]
        
        # Defining tensors for TorchScript
        u_out = torch.zeros((F.shape[0], F.shape[1], self.N_rank_ett[-1]), dtype=F.dtype,
            device=F.device) # temporary verctor output of etn
            
        
        data[_keys.NODE_FEATURES_ETN] = torch.zeros_like(F, dtype=F.dtype,
            device=F.device) # final feature output
        
        slices = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY].slices() # slices over irreps

        # getting w3j in dense mode
        w3j_dense = (
            self.w3j.to_dense()
            .reshape(self.w3j_shape)
            .contiguous()
        )   
        
        # First transform using second order tensors
        for i, slice in enumerate(slices):
            u_out[:, slice, :] = torch.einsum('ij,Nmj->Nmi', self.cores[-1][i].squeeze(-1), F[:, slice, :])

        # Series third order tensors
        for i in range(self.d - 2, 0, -1):

            # big contruction
            u_out = self.tps[i-1](u_out, F, w3j_dense, self.cores[i])

        # Last transform using second order tensor
        for i, slice in enumerate(slices):
            data[_keys.NODE_FEATURES_ETN][:, slice, :] = torch.einsum('ij,Nmj->Nmi', self.cores[0][i].squeeze(0), u_out[:, slice, :])
        
        
        # Reduction to scalar
        data[self.out_field] = (( data[_keys.NODE_FEATURES_ETN] * F ).sum(dim = (-2, -1) )).unsqueeze(-1)
        

        return data