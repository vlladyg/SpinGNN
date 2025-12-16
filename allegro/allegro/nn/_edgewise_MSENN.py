"""SpinGNN++ Energy Sum Modules.

This module contains the energy computation layers for SpinGNN++. These layers
take the equivariant features from MSENN/TENN and compute the various energy
contributions by contracting with spin vectors and basis matrices.

=== ENERGY CONTRIBUTIONS IN SPINGNN++ ===

The total magnetic energy is decomposed as:

    E = E_pair + E_BQ + E_J + E_A + E_TENN

1. E_pair: Standard pair energy (from EdgewiseEnergySum in _edgewise.py)
   - Computed from scalar edge features

2. E_BQ (Biquadratic): K_ij * (S_i · S_j)²
   - EdgewiseEnergySumBQ
   - Uses EDGE_K (scalar biquadratic coupling) from MLP on latent features

3. E_J (Exchange Tensor): S_i^T J_ij S_j
   - EdgewiseEnergySumJ
   - J_ij = Σ_k features_k * matrix_terms_J[k]
   - Contracts 9-component features with 9 basis matrices

4. E_A (On-site Anisotropy): S_i^T A_i S_i
   - EdgewiseEnergySumA
   - A_i = Σ_k (Σ_j features_ij) * matrix_terms_A[k]
   - First sums over neighbors j, then contracts with 6 basis matrices

5. E_TENN: Higher-order contribution
   - EdgewiseEnergySumTENN
   - Simple MLP on TENN latent features

=== ATOMWISE REDUCE ===

AtomwiseReduceSpinGNNPlus combines all 5 energy terms with learnable weights,
allowing the model to balance different physical contributions.

Authors: Vladimir Ladygin
"""

from typing import Optional
import math

import torch
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from .. import _keys
from .. import matrix_terms_J, matrix_terms_A


class EdgewiseReduce(GraphModuleMixin, torch.nn.Module):
    """Like ``nequip.nn.AtomwiseReduce``, but accumulating per-edge data into per-atom data."""

    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        normalize_edge_reduce: bool = True,
        avg_num_neighbors: Optional[float] = None,
        reduce="sum",
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        assert reduce in ("sum", "mean", "min", "max")
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field]}
            if self.field in irreps_in
            else {},
        )
        self._factor = None
        if normalize_edge_reduce and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get destination nodes 🚂
        edge_dst = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        out = scatter(
            data[self.field],
            edge_dst,
            dim=0,
            dim_size=len(data[AtomicDataDict.POSITIONS_KEY]),
            reduce=self.reduce,
        )

        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            out = out * factor

        data[self.out_field] = out

        return data


class EdgewiseEnergySumBQ(GraphModuleMixin, torch.nn.Module):
    """Compute biquadratic exchange energy: E_BQ = K_ij * (S_i · S_j)².
    
    Biquadratic exchange is a higher-order spin interaction that arises from
    quantum mechanical effects beyond the Heisenberg model. It favors either
    parallel or perpendicular spin alignment depending on the sign of K.
    
    === PHYSICS ===
    
    The biquadratic term: H_BQ = Σ_{ij} K_ij (S_i · S_j)²
    
    - K > 0: favors perpendicular spins (90° alignment)
    - K < 0: favors collinear spins (0° or 180°)
    
    This term is important for:
    - Spin-1 systems (where it arises naturally)
    - Itinerant magnets with strong spin fluctuations
    - Systems near magnetic phase transitions
    
    === COMPUTATION ===
    
    1. K_ij is predicted from EDGE_K (scalar MLP output)
    2. (S_i · S_j)² is computed from EDGE_SPIN_DISTANCE squared
    3. Per-edge energy: E_ij = K_ij * (S_i · S_j)²
    4. Scatter sum to per-atom energies
    """

    _factor: Optional[float]

    def __init__(
        self,
        num_types: int,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        per_edge_species_scale: bool = False,
        irreps_in={},
    ):
        """Initialize biquadratic energy sum module."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={_keys.EDGE_ENERGY_BQ: "0e", _keys.PER_ATOM_ENERGY_BQ: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        
        
        edge_eng_BQ = data[_keys.EDGE_K] * (data[_keys.EDGE_SPIN_DISTANCE] * data[_keys.EDGE_SPIN_DISTANCE]).unsqueeze(-1)
        data[_keys.EDGE_ENERGY_BQ] = edge_eng_BQ
        
        atom_eng = scatter(edge_eng_BQ, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[_keys.PER_ATOM_ENERGY_BQ] = atom_eng

        return data
    
    
class EdgewiseEnergySumJ(GraphModuleMixin, torch.nn.Module):
    """Compute exchange tensor energy: E_J = S_i^T J_ij S_j.
    
    This module computes the general bilinear spin interaction where J_ij is
    a full 3x3 exchange tensor that can encode:
    
    1. Heisenberg exchange (isotropic, L=0): J * (S_i · S_j)
    2. Dzyaloshinskii-Moriya interaction (antisymmetric, L=1): D · (S_i × S_j)
    3. Anisotropic symmetric exchange (traceless symmetric, L=2)
    
    === TENSOR CONSTRUCTION ===
    
    J_ij = Σ_k features_J[k] * matrix_terms_J[k]
    
    where:
    - features_J: 9-component output from MSENN (0e + 1e + 2e)
    - matrix_terms_J: 9 basis matrices from l2_matrix.py
      [H, DM1, DM2, DM3, ASEI1, ASEI2, ASEI3, ASEI4, ASEI5]
    
    === PHYSICS ===
    
    This decomposition allows learning arbitrary exchange tensors while
    respecting the irreducible representation structure:
    - The scalar (0e) gives isotropic Heisenberg coupling
    - The vector (1e) gives the DM vector
    - The tensor (2e) gives anisotropic exchange
    
    === COMPUTATION ===
    
    1. Contract features with basis matrices: J_ij[a,b] = Σ_k f_k * M_k[a,b]
    2. Compute bilinear form: E_ij = S_i^T J_ij S_j
    3. Scatter sum to per-atom energies
    """

    _factor: Optional[float]

    def __init__(
        self,
        num_types: int,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        per_edge_species_scale: bool = False,
        irreps_in={},
    ):
        """Initialize exchange tensor energy sum module."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={_keys.EDGE_ENERGY_J: "0e", _keys.PER_ATOM_ENERGY_J: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        
        node_spin = data[AtomicDataDict.SPIN_KEY]
        
        J_mtx = torch.einsum('ik,klm->ilm', data[_keys.EDGE_FEATURES_MSENN_J], matrix_terms_J)
        
        edge_eng_J = torch.einsum('ia,iab,ib->i', node_spin[edge_center], J_mtx, node_spin[edge_neighbor]).unsqueeze(-1)
        data[_keys.EDGE_ENERGY_J] = edge_eng_J
        
        atom_eng = scatter(edge_eng_J, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[_keys.PER_ATOM_ENERGY_J] = atom_eng

        return data
    
class EdgewiseEnergySumA(GraphModuleMixin, torch.nn.Module):
    """Compute on-site anisotropy energy: E_A = S_i^T A_i S_i.
    
    On-site anisotropy represents the energy cost of rotating a spin away
    from preferred crystallographic directions. This is a single-site term
    that depends on the local chemical environment.
    
    === PHYSICS ===
    
    The anisotropy tensor A_i must be symmetric (from energy minimization).
    It encodes:
    - Easy-axis anisotropy: energy minimum along a specific direction
    - Easy-plane anisotropy: energy minimum in a plane
    - Cubic anisotropy: multiple equivalent easy axes
    
    Typical forms:
    - Uniaxial: A = diag(0, 0, K) gives E = K * S_z²
    - Biaxial: A = diag(K1, K2, 0) gives E = K1*S_x² + K2*S_y²
    
    === TENSOR CONSTRUCTION ===
    
    Unlike J_ij (per-edge), A_i is per-atom. We compute it by:
    
    1. Get per-edge A features from MSENN (6 components: 0e + 2e, no 1e)
    2. Sum over neighbors: A_i = Σ_j A_ij (environment averaging)
    3. Contract with basis: A_i[a,b] = Σ_k f_k * matrix_terms_A[k][a,b]
    
    Note: No 1e (antisymmetric) component because A must be symmetric.
    
    === COMPUTATION ===
    
    1. Construct A_i from edge features (scatter sum)
    2. Compute quadratic form: E_i = S_i^T A_i S_i
    """

    _factor: Optional[float]

    def __init__(
        self,
        num_types: int,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        per_edge_species_scale: bool = False,
        irreps_in={},
    ):
        """Initialize on-site anisotropy energy sum module."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={_keys.EDGE_ENERGY_J: "0e", _keys.PER_ATOM_ENERGY_A: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        
        node_spin= data[AtomicDataDict.SPIN_KEY]
        
        
        A_mtx = torch.einsum('ik,klm->ilm', data[_keys.EDGE_FEATURES_MSENN_A].squeeze(-2), matrix_terms_A)
        
        A_mtx = scatter(A_mtx, edge_center, dim=0, dim_size=len(species))
        
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            A_mtx = A_mtx * factor

        atom_eng_A = torch.einsum('ia,iab,ib->i', node_spin, A_mtx, node_spin).unsqueeze(-1)
        data[_keys.PER_ATOM_ENERGY_A] = atom_eng_A

        return data


class EdgewiseEnergySumTENN(GraphModuleMixin, torch.nn.Module):
    """Sum TENN edgewise energies to per-atom contributions.
    
    This module handles the energy contribution from the Time-reversal
    Equivariant Tensor Network (TENN). Unlike the explicit J, A, K terms,
    the TENN energy captures higher-order spin-lattice coupling effects
    that don't fit neatly into the standard Hamiltonian form.
    
    === PHYSICS ===
    
    TENN can learn:
    - Higher-order exchange (4-spin, 6-spin terms)
    - Complex spin textures (skyrmions, spin spirals)
    - Spin-phonon coupling effects
    - Non-perturbative magnetic interactions
    
    The TENN branch processes combined position and spin information with
    proper time-reversal equivariance, then outputs a scalar energy.
    
    === COMPUTATION ===
    
    1. EDGE_ENERGY_TENN comes from MLP on TENN latent features
    2. Optional per-species-pair scaling
    3. Scatter sum to per-atom energies
    
    This is structurally similar to the standard EdgewiseEnergySum but
    operates on the TENN energy field instead of the MSENN energy field.
    """

    _factor: Optional[float]

    def __init__(
        self,
        num_types: int,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_energy_sum: bool = True,
        per_edge_species_scale: bool = False,
        irreps_in={},
    ):
        """Initialize TENN energy sum module."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={_keys.EDGE_ENERGY_TENN: "0e"},
            irreps_out={_keys.PER_ATOM_ENERGY_TENN: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        self.per_edge_species_scale = per_edge_species_scale
        if self.per_edge_species_scale:
            self.per_edge_scales_TENN = torch.nn.Parameter(torch.ones(num_types, num_types))
        else:
            self.register_buffer("per_edge_scales_TENN", torch.Tensor())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_eng = data[_keys.EDGE_ENERGY_TENN]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        center_species = species[edge_center]
        neighbor_species = species[edge_neighbor]

        if self.per_edge_species_scale:
            edge_eng = edge_eng * self.per_edge_scales_TENN[
                center_species, neighbor_species
            ].unsqueeze(-1)

        atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[_keys.PER_ATOM_ENERGY_TENN] = atom_eng

        return data
    
    
class AtomwiseReduceSpinGNNPlus(GraphModuleMixin, torch.nn.Module):
    """Combine all SpinGNN++ energy contributions into total energy.
    
    This module sums the 5 per-atom energy contributions from SpinGNN++
    with learnable per-contribution scaling factors:
    
    E_total = w1*E_pair + w2*E_BQ + w3*E_J + w4*E_A + w5*E_TENN
    
    === ENERGY CONTRIBUTIONS ===
    
    1. E_pair (field_eng): Standard pair energy from MSENN latent
    2. E_BQ (field_BQ): Biquadratic exchange K(S·S)²
    3. E_J (field_J): Exchange tensor S^T J S
    4. E_A (field_A): On-site anisotropy S^T A S
    5. E_TENN (field_TENN): Higher-order TENN contribution
    
    === LEARNABLE SCALES ===
    
    When per_contrib_scales=True (default), the module learns 5 scaling
    factors (per_contrib_scales_SpinGNNPlus) that balance the contributions.
    This is useful because:
    
    - Different terms may have different natural magnitudes
    - The network may need to emphasize certain physics
    - Helps with training stability
    
    === REDUCTION ===
    
    Supports:
    - "sum": simple sum over atoms
    - "mean": average over atoms
    - "normalized_sum": sum divided by sqrt(N_atoms)
    """
    constant: float

    def __init__(
        self,
        field_eng: str,
        field_BQ: str,
        field_J: str,
        field_A: str,
        field_TENN: str,
        out_field: Optional[str] = None,
        reduce="sum",
        avg_num_atoms=None,
        irreps_in={},
        per_contrib_scales: bool = True,
    ):
        """Initialize SpinGNN++ atomwise reduction module.
        
        Args:
            field_eng: Field name for pair energy
            field_BQ: Field name for biquadratic energy
            field_J: Field name for exchange tensor energy
            field_A: Field name for anisotropy energy
            field_TENN: Field name for TENN energy
            out_field: Output field name (default: reduce_field_eng)
            reduce: Reduction method ("sum", "mean", "normalized_sum")
            avg_num_atoms: Average atoms per structure (for normalized_sum)
            irreps_in: Input irreps specification
            per_contrib_scales: Whether to learn per-contribution scales
        """
        super().__init__()
        assert reduce in ("sum", "mean", "normalized_sum")
        self.constant = 1.0
        if reduce == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            reduce = "sum"
        self.reduce = reduce
        self.field_eng = field_eng
        self.field_BQ = field_BQ
        self.field_J = field_J
        self.field_A = field_A
        self.field_TENN = field_TENN
        self.out_field = f"{reduce}_{field_eng}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field_eng]}
            if self.field_eng in irreps_in
            else {},
        )

        self.per_contrib_scales = per_contrib_scales
        if self.per_contrib_scales:
            self.per_contrib_scales_SpinGNNPlus = torch.nn.Parameter(torch.ones(5))
        else:
            self.register_buffer("per_contrib_scales_SpinGNNPlus", torch.Tensor())
    

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_batch(data)

        term_eng = scatter(
            data[self.field_eng], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )
        term_BQ = scatter(
            data[self.field_BQ], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )
        term_J = scatter(
            data[self.field_J], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )

        term_A = scatter(
            data[self.field_A], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )
        
        term_TENN = scatter(
            data[self.field_TENN], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )

        if self.per_contrib_scales:
            for i, el in enumerate([term_eng, term_BQ, term_J, term_A, term_TENN]):
                el *= self.per_contrib_scales_SpinGNNPlus[i]
        
        data[self.out_field] = term_eng + term_BQ + term_J + term_A + term_TENN
        
        if self.constant != 1.0:
            data[self.out_field] = data[self.out_field] * self.constant
        return data