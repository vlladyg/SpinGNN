from typing import Optional, List
import math
import functools

import torch
from torch_runstats.scatter import scatter

from e3nn import o3
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.utils.tp_utils import tp_path_exists

from ._fc import ScalarMLPFunction
from .. import _keys

from ._strided import Contracter, MakeWeightedChannels, Linear
from .cutoffs import cosine_cutoff, polynomial_cutoff

@compile_mode("script")
class ETN_Module(nn.Module, GraphModuleMixin):
    def __init__(self,
                 d: int,
                 Nc: List[int],
                 N_rank_ett: List[int], 
                 irreps_in=None,
                 out_field: str = _keys.PER_ATOM_ENERGY_ETN):
        
        super().__init__()
        self.out_field = out_field
        
        self.register_buffer("N_rank_ett", torch.ad_tensor([1] + N_rank_ett + [1], 
                                                               dtype = torch.long))
        
        self.d = d
        self.Nc = irreps_in[_keys.NODE_FEATURES_F][0][0]
        
        # set up irreps
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                _keys.NODE_FEATURES_F
            ],
            irreps_out={out_field: o3.Irreps(
                    [(N_rank_ett[0], (0, 1))])}
        )
        
        
        
        # Parameters of the network
        
        # tensors for atomic features encoding
        lmax = irreps_in[_keys.EDGE_FEATURES_F].lmax # maximum spherical harmonic
        
        
        # Second order cores(first and last)
        core2_1 = torch.nn.Parameter((lmax, Nc[0], N_rank_ett[0]))
        core2_d = torch.nn.Parameter((lmax, N_rank_ett[-1], Nc[-1]))
        
        self.cores = [core2_1, core2_d]
        
        self.reset_parameters()
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
       

        # Get flattened square matrix of pair types
        pass

        return data
    
    def reset_parameters(self):
        for core in self.cores:
            torch.nn.init.kaiming_uniform_(core, a=math.sqrt(3))
        pass