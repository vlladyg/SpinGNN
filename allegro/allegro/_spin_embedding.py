"""Spin Embedding Modules for SpinGNN/SpinGNN++.

This module provides the spin-related embedding layers that extend the standard
NequIP/Allegro edge embeddings to include spin degree of freedom information.

=== OVERVIEW ===

In magnetic systems, the total energy depends on both atomic positions AND spin
orientations. This module provides:

1. `with_edge_spin_length`: Computes spin invariants (lengths and dot products)
2. `RadialBasisSpinDistanceEncoding`: Embeds spin dot products using radial basis
3. `SphericalHarmonicEdgeAttrsTENN`: Creates TENN-compatible edge attributes from
   both position vectors AND spin vectors

=== TIME-REVERSAL EQUIVARIANCE (TENN) ===

The key innovation for spin systems is handling time-reversal symmetry. Under time
reversal T:
- Positions r -> r (invariant)
- Spins S -> -S (flip sign)

This is encoded in e3nn using extended irreps (l, p, t) where:
- l: angular momentum
- p: parity under spatial inversion (+1 or -1)
- t: behavior under time reversal (+1 or -1)

For magnetic properties, we use t=-1 irreps to ensure the network respects
that reversing all spins changes the sign of magnetic interactions.

=== USAGE IN SPINGNN++ ===

The SpinGNN++ model uses these embeddings in sequence:
1. Standard radial basis for spatial distances
2. RadialBasisSpinDistanceEncoding for spin similarity (S_i · S_j)
3. SphericalHarmonicEdgeAttrsTENN combines 3 channels:
   - Edge direction Y(r_ij)
   - Center spin direction Y(S_i)  
   - Neighbor spin direction Y(S_j)

Authors: Vladimir Ladygin
"""

from typing import Dict, Any

import torch
import torch.jit

from typing import Union

from e3nn.util.jit import compile_mode
from e3nn import o3

# Make the keys available in this module
from ._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from . import _keys
from nequip.data import AtomicDataDict
from nequip.nn.radial_basis import BesselBasis

from nequip.nn import GraphModuleMixin


# Define a type alias
Type = Dict[str, torch.Tensor]



@torch.jit.script
def with_edge_spin_length(data: Type, with_distance: bool = True) -> Type:
    """Compute spin invariants for each atom and edge in the graph.

    This function computes the spin-related quantities needed for SpinGNN:
    
    1. NODE_SPIN_LENGTH: |S_i| for each atom (scalar spin magnitude)
    2. NODE_SPIN_VEC: S_i / |S_i| (unit spin vector)
    3. EDGE_SPIN_DISTANCE: (S_i · S_j) / (|S_i| |S_j|) (cosine similarity)
    
    The spin distance is a key invariant for the spin Hamiltonian:
    - Heisenberg term: J * (S_i · S_j) 
    - Biquadratic term: K * (S_i · S_j)²
    
    This function is JIT-scriptable for use in TorchScript models.

    Args:
        data: AtomicDataDict containing NODE_SPIN field [n_atoms, 3]
        with_distance: If True, compute edge spin distances. If False, only
                       compute node spin lengths.

    Returns:
        Updated data dict with:
        - NODE_SPIN_LENGTH: [n_atoms] spin magnitudes
        - NODE_SPIN_VEC: [n_atoms, 3] unit spin vectors (if with_distance=True)
        - EDGE_SPIN_DISTANCE: [n_edges] spin dot products (if with_distance=True)
    """
    # Build node spin norms
    
    if not _keys.NODE_SPIN_LENGTH in data:
        data[_keys.NODE_SPIN_LENGTH] = torch.linalg.norm(data[AtomicDataDict.SPIN_KEY], dim=-1)
        

    # Build spin distance
    if with_distance and _keys.EDGE_SPIN_DISTANCE not in data:
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        unit_spin = data[AtomicDataDict.SPIN_KEY]/data[_keys.NODE_SPIN_LENGTH][:, None]
        data[_keys.NODE_SPIN_VEC] = unit_spin
        
        
        data[_keys.EDGE_SPIN_DISTANCE] = torch.einsum("ki,kj->k", unit_spin[edge_index[1]], unit_spin[edge_index[0]])
        
    return data


@compile_mode("script")
class RadialBasisSpinDistanceEncoding(GraphModuleMixin, torch.nn.Module):
    """Encode the spin-spin dot product using a radial basis expansion.
    
    Similar to how RadialBasisEdgeEncoding encodes spatial distances r_ij,
    this module encodes spin distances (S_i · S_j) using a radial basis.
    
    The spin distance ranges from -1 (antiparallel) to +1 (parallel), so
    the basis functions should be defined over this domain.
    
    This encoding allows the network to learn smooth functions of the
    spin alignment, which is crucial for capturing:
    - Ferromagnetic preference (low energy when parallel)
    - Antiferromagnetic preference (low energy when antiparallel)
    - Spin spiral phases (complex angular dependence)
    
    The output is a set of scalar invariants that can be fed into latent MLPs.
    
    Output irreps: (num_basis × 0e) for spin distance embedding
                   + (1 × 0e) for spin length
    """
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        basis_kwargs={},
        out_field: str = _keys.EDGE_SPIN_DISTANCE_EMBEDDING,
        irreps_in=None,
    ):
        """Initialize the spin distance encoder.
        
        Args:
            basis: Radial basis class (default: BesselBasis)
            basis_kwargs: Arguments passed to basis constructor
            out_field: Output field name for embedded spin distances
            irreps_in: Input irreps specification
        """
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))]), 
                        _keys.NODE_SPIN_LENGTH: o3.Irreps([(1, (0, 1))])},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_spin_length(data, with_distance=True)
        edge_spin_distance = data[_keys.EDGE_SPIN_DISTANCE]
        edge_spin_distance_embedded = (
            self.basis(edge_spin_distance) #* self.cutoff(edge_length)[:, None]
        )
        data[self.out_field] = edge_spin_distance_embedded
        return data

    
    
@compile_mode("script")
class SphericalHarmonicEdgeAttrsTENN(GraphModuleMixin, torch.nn.Module):
    """Construct TENN edge attributes from edge vectors AND spin vectors.
    
    === EXTENSION FROM STANDARD SPHERICAL HARMONIC EDGE ATTRS ===
    
    The standard SphericalHarmonicEdgeAttrs only uses edge vectors r_ij.
    This TENN version creates a 3-channel edge attribute combining:
    
    1. Y(r_ij): Spherical harmonics of edge direction
    2. Y(S_i): Spherical harmonics of center atom spin direction  
    3. Y(S_j): Spherical harmonics of neighbor atom spin direction
    
    === TIME-REVERSAL SYMMETRY ===
    
    The spherical harmonics are computed with time_reversal=True and parity=False,
    meaning they transform under the extended O(3) × Z_2^T group. The irreps have
    signature (l, p, t) where t=-1 indicates odd behavior under time reversal.
    
    This is essential for magnetic systems because:
    - Spin vectors S flip sign under time reversal: T(S) = -S
    - Energy must be invariant under time reversal
    - Magnetic interactions (exchange, DM, etc.) have specific T-symmetry
    
    === OUTPUT STRUCTURE ===
    
    The output has shape [n_edges, 3, dim_sh] where:
    - First channel: edge direction spherical harmonics
    - Second channel: center spin spherical harmonics
    - Third channel: neighbor spin spherical harmonics
    
    This is then processed by MakeWeightedChannelsTENN which learns to mix
    these three geometric channels with learnable weights.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh_TENN (int, str, or o3.Irreps): if int, will be treated as 
            lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY): output data field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh_TENN: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        #out_field: str = 'edge_attr_tmp',
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field

        # Should only be applied to noncoliear setting
        #assert data[AtomicDataDict.SPIN_KEY].shape[-1] == 3 and len(data[AtomicDataDict.SPIN_KEY].shape) > 1
        
        if isinstance(irreps_edge_sh_TENN, int):
            self.irreps_edge_sh_TENN = o3.Irreps.spherical_harmonics(irreps_edge_sh_TENN)
        else:
            self.irreps_edge_sh_TENN = o3.Irreps(irreps_edge_sh_TENN)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: o3.Irreps([(3, ir) for _, ir in self.irreps_edge_sh_TENN]).sort().irreps},
        )
        self.sh_edge_vec = o3.SphericalHarmonics(
            self.irreps_edge_sh_TENN, edge_sh_normalize, edge_sh_normalization, 
            time_reversal=True, parity = False
        )
        self.sh_node_spin_vec = o3.SphericalHarmonics(
            self.irreps_edge_sh_TENN, edge_sh_normalize, edge_sh_normalization, 
            time_reversal=True, parity = False
        )
        

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        #data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        node_spin_vec = data[_keys.NODE_SPIN_VEC]
        
        edge_sh = self.sh_edge_vec(edge_vec).unsqueeze(-2)
        
        edge_node_spin = self.sh_node_spin_vec(node_spin_vec).unsqueeze(-2)
        
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_node_center = edge_node_spin[edge_index[1]]
        edge_node_neighbor = edge_node_spin[edge_index[0]]
        
        
        # calculating ind for coping (redundant not need)
        #i_shift_0 = 0
        #i_shift_1 = 1
        #i_shift_2 = 2
        #ind_0 = []
        #ind_1 = []
        #ind_2 = []
        #for l in range(self.irreps_edge_sh_TENN.lmax + 1):
        #    ind_0 += list(range(i_shift_0, i_shift_0 + 2*l+1))
        #    ind_1 += list(range(i_shift_1, i_shift_1 + 2*l+1))
        #    ind_2 += list(range(i_shift_2, i_shift_2 + 2*l+1))

        #    i_shift_0 += 3*(2*l+1)
        #    i_shift_1 = i_shift_0 + 2*(l+1) + 1
        #    i_shift_2 = i_shift_1 + 2*(l+1) + 1
        
        #ind_0 = torch.tensor(ind_0, dtype = torch.long)
        #ind_1 = torch.tensor(ind_1, dtype = torch.long)
        #ind_2 = torch.tensor(ind_2, dtype = torch.long)
        
        #data[self.out_field] = torch.zeros((edge_sh.shape[0], self.irreps_out[self.out_field].dim))
        
        #print(edge_sh.shape)
        #print(data[self.out_field].shape)
        #data[self.out_field] = torch.index_copy(data[self.out_field], -1, ind_0, edge_sh)
        #data[self.out_field] = torch.index_copy(data[self.out_field], -1, ind_1, edge_node_center)
        #data[self.out_field] = torch.index_copy(data[self.out_field], -1, ind_2, edge_node_neighbor)
        
        data[self.out_field] = torch.concat([edge_sh, edge_node_center, edge_node_neighbor], dim = -2)
        
        return data