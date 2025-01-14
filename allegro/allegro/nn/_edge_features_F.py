from typing import Optional
import math

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
        self.A = torch.nn.Parameter(torch.Tensor(lmax + 1, N_rank_spec, num_types**2))
        self.B =  torch.nn.Parameter(torch.Tensor(lmax + 1, Nc, num_basis, N_rank_spec))
        
        
        
        self.reset_parameters()
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
       

        # Get flattened square matrix of pair types
        atom_types_embed = data[_keys.EDGE_TYPE_KEY]
    
        
        # Algo from ETN paper to gen F (notation preserved)
        Q = data[AtomicDataDict.EDGE_EMBEDDING_KEY]
        a = self.A[:, :, atom_types_embed].squeeze(-1)
        b = torch.einsum('Lrnk,LkE,En->ELr', self.B, a, Q)

        Y = data[AtomicDataDict.EDGE_ATTRS_KEY]
    
        F = torch.concat([torch.einsum('Em,En->Emn', Y[:, slices],
                                       b[:, l]) for l, slices in enumerate(self.irreps_edge_sh.slices())], dim = -2)

        
        # Saving to data
        data[_keys.EDGE_FEATURES_F] = F
        
        # Passing massage to the node creating node feature
        #data['node_attrs'] = scatter(F, edge_center, dim=0, dim_size=len(species))

        return data
    
    def reset_parameters(self):
        #torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(3))
        torch.nn.init.kaiming_uniform_(self.B, a=math.sqrt(3))
        pass