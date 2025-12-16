"""Equivariant Tensor Network (ETN) Module.

This module implements the ETN architecture, a fundamentally different approach
from Allegro that uses tensor train decomposition with Wigner-3j symbols.

=== ETN vs ALLEGRO ===

While Allegro uses:
- Message passing with learned pair interactions
- Edge-based representations
- Latent MLPs + tensor products

ETN uses:
- Tensor train (TT) decomposition of high-order equivariant tensors
- Node-based representations built from edge features
- Direct contraction with Wigner-3j coupling coefficients

=== TENSOR TRAIN STRUCTURE ===

ETN parameterizes a high-order equivariant tensor using the tensor train format:

    T = core2_1 × core3[0] × core3[1] × ... × core3[d-3] × core2_d

where:
- core2_1, core2_d: Second-order boundary cores (matrices per l)
- core3[i]: Third-order cores with Wigner-3j coupling

The depth 'd' controls the body-order of interactions:
- d=2: 2-body (pair interactions only)
- d=3: 3-body
- d=4: 4-body
- etc.

=== WIGNER-3J COUPLING ===

The third-order cores use Wigner-3j symbols to couple angular momenta:

    C^{l1,l2,l3}_{m1,m2,m3} = <l1,m1; l2,m2 | l3,m3>

This ensures proper SO(3) equivariance of the network.

=== COMPUTATION FLOW ===

1. Input: Node features F from EdgeFeatures_F (aggregated from edges)
2. Apply boundary core: u = core2_d × F
3. For each intermediate core (right to left):
   - Contract with Wigner-3j: T_2 = Σ_{l3} w3j × core3 × u
   - Contract with features: u_new = Σ_{l2} T_2 × F
4. Apply final boundary core: output = core2_1 × u
5. Reduce to scalar energy: E = (output * F).sum()

Authors: Vladimir Ladygin
"""

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

from ._strided import Contracter, MakeWeightedChannels, Linear
from .cutoffs import cosine_cutoff, polynomial_cutoff
from e3nn.o3 import wigner_3j


def tri_ineq(l1, l2, l3):
    """Check triangular inequality for angular momentum coupling.
    
    Returns True if (l1, l2, l3) can couple according to angular momentum
    addition rules: |l1 - l2| <= l3 <= l1 + l2
    """
    return max([l1, l2, l3]) <= min([l1 + l2, l2 + l3, l1 + l3])


@compile_mode("script")
class ETN_Module(nn.Module, GraphModuleMixin):
    """Equivariant Tensor Network module for atomic energy prediction.
    
    This module implements the core ETN computation using tensor train
    decomposition with Wigner-3j coupling.
    
    === PARAMETERS ===
    
    The network has the following learnable parameters:
    
    1. core2_1: [lmax+1, Nc, N_rank[0]] - First boundary core
       Maps from feature channels to first TT rank
       
    2. core2_d: [lmax+1, N_rank[-1], Nc] - Last boundary core
       Maps from last TT rank back to feature channels
       
    3. cores3: List of dicts mapping (l1,l2,l3) -> [N_rank[i], Nc, N_rank[i+1]]
       Third-order cores for each valid angular momentum triple
    
    === TENSOR TRAIN RANKS ===
    
    N_rank_ett controls the expressivity:
    - Higher ranks = more parameters = more expressive
    - Typical values: [4, 8, 16, 8, 4] for d=5
    
    === BODY ORDER ===
    
    The depth 'd' determines the body order:
    - d=2: Only boundary cores, 2-body interactions
    - d=3: One intermediate core, 3-body
    - d=4: Two intermediate cores, 4-body
    - etc.
    
    Higher body orders capture more complex many-body effects but are
    more expensive to compute.
    """
    def __init__(self,
                 d: int,
                 N_rank_ett: List[int], 
                 irreps_in=None,
                 out_field: str = AtomicDataDict.PER_ATOM_ENERGY_KEY):
        """Initialize ETN module.
        
        Args:
            d: Depth of tensor train (body order = d)
            N_rank_ett: List of TT ranks [r0, r1, ..., r_{d-2}]
            irreps_in: Input irreps specification
            out_field: Output field for per-atom energy
        """
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
        self.core2_1 = torch.nn.Parameter(torch.Tensor(lmax+1, self.Nc, N_rank_ett[0]))
        self.core2_d = torch.nn.Parameter(torch.Tensor(lmax+1, N_rank_ett[-1], self.Nc))
        
        
        
        # Third order cores
        
        # all wigner3j 
        self.w3j_big = [[[wigner_3j(l1, l2, l3) if tri_ineq(l1, l2, l3) else None for l3 in range (lmax+1)] for l2 in range(lmax+1)] for l1 in range(lmax+1)]
        
        # all third order free parameters
        self.cores3 = [{(l1, l2, l3): torch.nn.Parameter(torch.Tensor(N_rank_ett[r], self.Nc, N_rank_ett[r+1])) for l1 in range (lmax+1) for l2 in range(lmax+1) for l3 in range(lmax+1) if tri_ineq(l1, l2, l3)} for r in range(d - 2)] 
        
        self.reset_parameters()
        
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        # Input features
        F = data[_keys.NODE_FEATURES_F]
        
        # Defining tensors for TorchScript
        u_out = torch.zeros((F.shape[0], F.shape[1], self.N_rank_ett[-1]), dtype=F.dtype,
            device=F.device) # temporary verctor output of etn
            
        
        data[_keys.NODE_FEATURES_ETN] = torch.zeros_like(F, dtype=F.dtype,
            device=F.device) # final feature output
        
        slices = self.irreps_in[AtomicDataDict.EDGE_ATTRS_KEY].slices() # slices over irreps
        
        # First transform using second order tensors
        for i, slice in enumerate(slices):
            u_out[:, slice, :] = torch.einsum('ij,Nmj->Nmi', self.core2_d[i], F[:, slice, :])
        
        # Series third order tensors
        for i in range(self.d - 2 - 1, -1, -1):
            
            # TODO: now define localy, mb define for all, to ensure computational graph
            T_2_tmp = [[torch.zeros(F.shape[0], 2*l1+1, 2*l2+1, self.N_rank_ett[i - 1], self.Nc, dtype=F.dtype, device=F.device) for l2 in range(self.lmax + 1)] for l1 in range(self.lmax + 1)] # result of first reduction of order 3 tensor
            
            # First contraction with previous feature vector
            for l1 in range(self.lmax + 1):
                for l2 in range(self.lmax + 1):
                    for l3, slice in enumerate(slices):
                        if tri_ineq(l1, l2, l3):
                            #T_3 = self.w3j_big[l1][l2][l3][..., None, None, None] * self.cores3[i][(l1, l2, l3)][None, None, None, ...]
                            T_2_tmp[l1][l2] += torch.einsum('abc,ijk,Nck->Nabij', self.w3j_big[l1][l2][l3], self.cores3[i][(l1, l2, l3)], u_out[:, slice, :])
            
            
            # Second contraction with F vector
            u_out_new = torch.zeros((F.shape[0], F.shape[1], self.N_rank_ett[i - 1]), dtype=F.dtype,
                            device=F.device) # temporary verctor output of etn
            
            for l1 in range(self.lmax + 1):    
                for l2, slice in enumerate(slices):
                    u_out_new[:, slices[l1], :] += torch.einsum('Nabij,Nbj->Nai', T_2_tmp[l1][l2], F[:, slice, :])
            
            
            u_out = u_out_new
            
        # Last transform using second order tensor
        for i, slice in enumerate(slices):
            data[_keys.NODE_FEATURES_ETN][:, slice, :] = torch.einsum('ij,Nmj->Nmi', self.core2_1[i], u_out[:, slice, :])
        
        
        # Reduction to scalar
        data[self.out_field] = (( data[_keys.NODE_FEATURES_ETN] * F ).sum(dim = (-2, -1) )).unsqueeze(-1)
        

        return data
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.core2_1, a=math.sqrt(3))
        torch.nn.init.kaiming_uniform_(self.core2_d, a=math.sqrt(3))
        
        for core in self.cores3:
            for key in core:
                torch.nn.init.kaiming_uniform_(core[key], a=math.sqrt(3))