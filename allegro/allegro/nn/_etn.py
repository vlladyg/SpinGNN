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


# Triangular ineguality for path existance
def tri_ineq(l1, l2, l3):
    return max([l1, l2, l3]) <= min([l1 + l2, l2 + l3, l1 + l3])


@compile_mode("script")
class ETN_Module(nn.Module, GraphModuleMixin):
    def __init__(self,
                 d: int,
                 N_rank_ett: List[int], 
                 irreps_in=None,
                 out_field: str = _keys.PER_ATOM_ENERGY_ETN):
        
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
                    [(N_rank_ett, ir) for _, ir in irreps_in[_keys.NODE_FEATURES_F] ]),
                        out_field: o3.Irreps([(1, (0, 1))])}
        
        
        
        # Parameters of the network
        
        # tensors for atomic features encoding
        lmax = irreps_in[_keys.EDGE_FEATURES_F].lmax # maximum spherical harmonic
        
        
        # Second order cores(first and last)
        core2_1 = torch.nn.Parameter((lmax, Nc, N_rank_ett[0]))
        core2_d = torch.nn.Parameter((lmax, N_rank_ett[-1], Nc))
        
        
        
        # Third order cores
        
        # all wigner3j 
        w3j_big = [[[wigner_3j(i, j, k) if tri_ineq(i, j, k) else None for i in range (lmax)] for j in range(lmax)] for k in range(lmax)]
        
        # all third order free parameters
        core3 = [ [[[torch.nn.Parameter((N_rank_ett[r], Nc, N_rank_ett[r+1])) if tri_ineq(i, j, k) else None for i in range (lmax)] for j in range(lmax)] for k in range(lmax)] for r in d - 2] 
        
        
        # Setting all parameters in one list
        self.cores = [core2_1] + core3 + [core2_d]
        
        self.reset_parameters()
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        # Input features
        F = data[_keys.NODE_FEATURES_F]
        
        # Defining tensors for TorchScript
        u_out = torch.zeros((F.shape[:-1], self.N_rank_ett[-1]), dtype=F.dtype,
            device=F.device) # temporary verctor output of etn
            
        
        data[_keys.NODE_FEATURES_ETN] = torch.zeros_like(F.shape, dtype=F.dtype,
            device=F.device) # final feature output
        
        slices = self.irreps[_keys.EDGE_ATTRS].slices() # slices over irreps
        
        # First transform using second order tensors
        for i, slice in enumerate(slices):
            u_out[:, slice, :] = torch.einsum('ij,Nmi->Nmj', self.cores[-1][i], F[:, slice, :])
        
        # Series third order tensors
        for i in range(self.d - 2, 0, -1):
            
            # TODO: now define localy, mb define for all, to ensure computational graph
            T_2_tmp = [[torch.zeros(F.shape[0], 2*l1+1, 2*l2+1, self.N_rank_ett[i - 1], self.Nc, dtype=F.dtype, device=F.device) for l1 in range(self.lmax + 1)] for l1 in range(self.lmax + 1) for l2 in range(self.lmax + 1)] # result of first reduction of order 3 tensor
            
            # First contraction with previous feature vector
            for l3, slice in enumerate(slices):
                for l1 in range(self.lmax + 1):
                    for l2 in range(self.lmax + 1):
                        if tri_ineq(l1, l2, l3):
                            T_2_tmp[l1][l2] += torch.einsum('abcijk,Nck->Nabij', T_3[l1][l2][l3], u_out[:, slice, :])
            
            
            # Second contraction with F vector
            u_out_new = torch.zeros((F.shape[:-1], self.N_rank_ett[i - 1]), dtype=F.dtype,
                            device=F.device) # temporary verctor output of etn
            
            for l1 in in range(self.lmax + 1):    
                for l2, slice in enumerate(slices):
                    u_out_new[:, slices[l1], :] += torch.einsum('Nabij,Nbj->Nai', T_2_tmp[l1][l2], F[:, slice, :])
            
            
            u_out = u_out_new
            
        # Last transform using second order tensor
        for i, slice in enumerate(slices):
            data[_keys.NODE_FEATURES_ETN][:, slice, :] = torch.einsum('ij,Nmi->Nmj', self.cores[0][i], u_out[:, slices, :])
        
        
        # Reduction to scalar
        data[self.out_field] = ( data[_keys.NODE_FEATURES_ETN] * F ).sum(dim = (-2, -1))
        

        return data
    
    def reset_parameters(self):
        for core in self.cores:
            torch.nn.init.kaiming_uniform_(core, a=math.sqrt(3))
        pass