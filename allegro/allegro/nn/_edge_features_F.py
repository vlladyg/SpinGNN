from typing import Optional
import math
from e3nn.util.codegen import CodeGenMixin

import torch
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from .. import _keys

from torch import nn
from e3nn import o3


class EdgeFeatures_F(nn.Module, GraphModuleMixin):
    def __init__(self,
                 num_types: int,
                 Nc: int, 
                 num_basis: int = 8, 
                 N_rank_spec: int = 4,
                 irreps_in=None,
                 out_field: str = _keys.EDGE_FEATURES_F):
        
        super().__init__()
        self.out_field = out_field
        
        self.irreps_edge_sh = irreps_in[AtomicDataDict.EDGE_ATTRS_KEY]
        
        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                AtomicDataDict.EDGE_ATTRS_KEY,
                AtomicDataDict.EDGE_EMBEDDING_KEY,
                _keys.EDGE_TYPE_KEY
            ],
            irreps_out={out_field: o3.Irreps([(Nc, ir) for _, ir in self.irreps_edge_sh])}
        )
        

        
        # parameters of the system
        self.num_types = num_types # number of types
        
        
        # Parameters of the network
        self.Nc = Nc # number of output features
        self.num_basis = num_basis # number of radial basis functions
        self.N_rank_spec = N_rank_spec # encoding of species after one_hot
        
        # tensors for atomic features encoding
        lmax = self.irreps_edge_sh.lmax # maximum spherical harmonic
        
        self._module = ScalarMLPFunction(
            lmax=lmax,
            num_types=self.num_types,
            Nc=self.Nc,
            num_basis=self.num_basis,
            N_rank_spec=self.N_rank_spec,
            irreps_edge_sh=self.irreps_edge_sh,
        )
        

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data[self.out_field] = self._module(data[AtomicDataDict.EDGE_EMBEDDING_KEY],
                                            data[_keys.EDGE_TYPE_KEY],
                                            data[AtomicDataDict.EDGE_ATTRS_KEY])
        
        return data


class EdgeFeatures_FFunction(CodeGenMixin, torch.nn.Module):
    """Module implementing an MLP according to provided options."""

    in_features: int
    out_features: int

    def __init__(
        self,
         lmax: int,
         num_types: int,
         Nc: int, 
         num_basis: int = 8, 
         N_rank_spec: int = 4,
         irreps_edge_sh: o3.Irreps = o3.Irreps("1x0e + 1x1o + 1x2e")
    ):
        super().__init__()

        
        # Code
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        Q = Proxy(graph.placeholder("x"))
        atom_types_embed = Proxy(graph.placeholder("z"))
        Y = Proxy(graph.placeholder("y"))
        F = Proxy(graph.placeholder("F"))
        norm_from_last: float = 1.0

        base = torch.nn.Module()

        # make weights
        w = torch.empty(h_in, h_out)
        w.normal_()

        A = torch.empty(lmax + 1, N_rank_spec, num_types**2)
        A.normal_()
        
        B =  torch.empty((lmax + 1, Nc, num_basis, N_rank_spec)           
        B.normal_()

        # generate code
        params[f"A"] = A
        A = Proxy(graph.get_attr(f"A"))

        params[f"B"] = B
        B = Proxy(graph.get_attr(f"B"))


        # Algo from ETN paper to gen F (notation preserved)
        a = A[:, :, atom_types_embed].squeeze(-1)
        b = torch.einsum('Lrnk,LkE,En->ELr', B, a, Q)
    
        F = torch.concat([torch.einsum('Em,En->Emn', Y[:, slices],
                                       b[:, l]) for l, slices in enumerate(irreps_edge_sh.slices())], dim = -2)

        graph.output(F.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))


        self._codegen_register({"_forward": fx.GraphModule(base, graph)})

    def forward(self, x, z, y):
        return self._forward(x, z, y)