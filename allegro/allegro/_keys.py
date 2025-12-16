"""Keys file to overcome TorchScript constants bug.

This module defines string constants for data dictionary keys used throughout the
SpinGNN/SpinGNN++ codebase. Using Final[str] type hints ensures TorchScript
compatibility for JIT compilation.

=== EXTENSIONS FROM ORIGINAL ALLEGRO ===

This file extends the original Allegro keys with additional fields for:

1. SPIN FIELDS (SpinGNN/SpinGNN++ additions):
   - NODE_SPIN_VEC: Unit spin vector per atom (normalized spin direction)
   - NODE_SPIN_LENGTH: Scalar spin magnitude per atom
   - EDGE_SPIN: Per-edge spin contribution for message passing
   - EDGE_SPIN_DISTANCE: Dot product between spin vectors (S_i · S_j)
   - EDGE_SPIN_DISTANCE_EMBEDDING: Radial basis encoding of spin distance

2. ENERGY CONTRIBUTION FIELDS (SpinGNN++ multi-term Hamiltonian):
   - EDGE_ENERGY: Standard pair energy (from original Allegro)
   - EDGE_ENERGY_BQ: Biquadratic exchange energy K(S_i · S_j)²
   - EDGE_ENERGY_J: Exchange tensor energy S_i^T J_ij S_j
   - EDGE_ENERGY_A: On-site anisotropy energy S_i^T A_i S_i
   - EDGE_ENERGY_TENN: Time-reversal equivariant tensor network energy
   - EDGE_ENERGY_HEGNN/SEGNN: Legacy Heisenberg/Steerable EGNN energies

3. FEATURE FIELDS (MSENN outputs for spin Hamiltonian tensors):
   - EDGE_FEATURES_MSENN_J: 9-component tensor for J_ij (0e+1e+2e irreps)
   - EDGE_FEATURES_MSENN_A: 6-component tensor for A_i (0e+2e irreps)

4. EXCHANGE COUPLING FIELDS:
   - EDGE_J: Heisenberg exchange coupling J_ij (scalar)
   - EDGE_K: Biquadratic exchange coupling K_ij (scalar)
"""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

from nequip.data import register_fields

<<<<<<< HEAD
# [n_edge, 1]: define edge atomic type Zij as Zi * num_types + Zj
EDGE_TYPE_KEY: Final[str] = "edge_types"
EDGE_SPIN: Final[str] = "edge_spin"
EDGE_ENERGY: Final[str] = "edge_energy"
EDGE_FEATURES: Final[str] = "edge_features"
EDGE_FEATURES_MSENN_J: Final[str] = "edge_features_MSENN_J"
EDGE_FEATURES_MSENN_A: Final[str] = "edge_features_MSENN_A"

    
EDGE_FEATURES_F: Final[str] = "edge_features_F"

    
PER_ATOM_ENERGY_HEGNN: Final[str] = "atomic_energy_HEGNN"
PER_ATOM_ENERGY_SEGNN: Final[str] = "atomic_energy_SEGNN"
PER_ATOM_ENERGY_BQ: Final[str] = "atomic_energy_BQ"
PER_ATOM_ENERGY_J: Final[str] = "atomic_energy_J"
PER_ATOM_ENERGY_A: Final[str] = "atomic_energy_A"
PER_ATOM_ENERGY_TENN: Final[str] = "atomic_energy_TENN"

    
NODE_FEATURES_F: Final[str] = "node_features_F"
NODE_FEATURES_ETN: Final[str] = "node_features_ETN"    
    
    
NODE_SPIN_VEC: Final[str] = "node_spin_vec"
NODE_SPIN_LENGTH: Final[str] = "node_spin_length"
EDGE_SPIN: Final[str] = "edge_spin"
EDGE_SPIN_DISTANCE: Final[str] = "edge_spin_distance"
EDGE_SPIN_DISTANCE_EMBEDDING: Final[str] = "edge_spin_distance_embdedding"

       
    
EDGE_J: Final[str] = "edge_J"
EDGE_K: Final[str] = "edge_K"
EDGE_ENERGY_HEGNN: Final[str] = "edge_energy_HEGNN"
EDGE_ENERGY_SEGNN: Final[str] = "edge_energy_SEGNN"
EDGE_ENERGY_BQ: Final[str] = "edge_energy_BQ"
EDGE_ENERGY_J: Final[str] = "edge_energy_J"
EDGE_ENERGY_A: Final[str] = "edge_energy_A"
EDGE_ENERGY_TENN: Final[str] = "edge_energy_TENN"    
    
register_fields(node_fields=[NODE_FEATURES_F, NODE_FEATURES_ETN, NODE_SPIN_LENGTH, NODE_SPIN_VEC])
register_fields(edge_fields=[EDGE_TYPE_KEY, EDGE_FEATURES_F, EDGE_ENERGY, EDGE_FEATURES, EDGE_SPIN, 
                             EDGE_SPIN_DISTANCE, EDGE_SPIN_DISTANCE_EMBEDDING, 
                             EDGE_J, EDGE_ENERGY_HEGNN, EDGE_ENERGY_SEGNN,
                             EDGE_K, EDGE_ENERGY_BQ,
                             EDGE_FEATURES_MSENN_J, EDGE_FEATURES_MSENN_A,
                             EDGE_ENERGY_J, EDGE_ENERGY_A,
                             EDGE_ENERGY_TENN])
register_fields(graph_fields=[PER_ATOM_ENERGY_HEGNN, PER_ATOM_ENERGY_SEGNN,
                              PER_ATOM_ENERGY_BQ, PER_ATOM_ENERGY_J, PER_ATOM_ENERGY_A,
                              PER_ATOM_ENERGY_TENN])
=======
# =============================================================================
# EDGE FEATURES AND ENERGIES (Original Allegro + Extensions)
# =============================================================================

# Original Allegro edge outputs
EDGE_ENERGY: Final[str] = "edge_energy"          # Standard pair energy from Allegro
EDGE_FEATURES: Final[str] = "edge_features"      # Latent edge features (invariants)

# MSENN (Magnetic Steerable E(3) NN) output features for spin Hamiltonian tensors
# These are equivariant features that get contracted with spin vectors
EDGE_FEATURES_MSENN_J: Final[str] = "edge_features_MSENN_J"  # 9 components: 0e+1e+2e for J_ij tensor
EDGE_FEATURES_MSENN_A: Final[str] = "edge_features_MSENN_A"  # 6 components: 0e+2e for A_i tensor

# =============================================================================
# PER-ATOM ENERGY CONTRIBUTIONS (SpinGNN++ multi-term Hamiltonian)
# =============================================================================

PER_ATOM_ENERGY_HEGNN: Final[str] = "atomic_energy_HEGNN"   # Heisenberg EGNN contribution (legacy)
PER_ATOM_ENERGY_SEGNN: Final[str] = "atomic_energy_SEGNN"   # Steerable EGNN contribution (legacy)
PER_ATOM_ENERGY_BQ: Final[str] = "atomic_energy_BQ"         # Biquadratic exchange: K(S_i·S_j)²
PER_ATOM_ENERGY_J: Final[str] = "atomic_energy_J"           # Exchange tensor: S_i^T J_ij S_j
PER_ATOM_ENERGY_A: Final[str] = "atomic_energy_A"           # On-site anisotropy: S_i^T A_i S_i
PER_ATOM_ENERGY_TENN: Final[str] = "atomic_energy_TENN"     # TENN (time-reversal equivariant) contribution
PER_ATOM_SPIN_KEY: Final[str] = "atomic_spin"               # Per-atom spin magnitude

# =============================================================================
# NODE (ATOM) SPIN FIELDS
# =============================================================================

NODE_SPIN_VEC: Final[str] = "node_spin_vec"      # Unit spin vector: S_i / |S_i| (3D vector per atom)
NODE_SPIN_LENGTH: Final[str] = "node_spin_length" # Spin magnitude: |S_i| (scalar per atom)

# =============================================================================
# EDGE SPIN FIELDS
# =============================================================================

EDGE_SPIN: Final[str] = "edge_spin"              # Edge spin contribution for message passing
EDGE_SPIN_DISTANCE: Final[str] = "edge_spin_distance"  # Spin dot product: S_i · S_j / (|S_i||S_j|)
EDGE_SPIN_DISTANCE_EMBEDDING: Final[str] = "edge_spin_distance_embdedding"  # Radial basis encoding of spin distance

# =============================================================================
# EXCHANGE COUPLING PARAMETERS
# =============================================================================

EDGE_J: Final[str] = "edge_J"  # Heisenberg exchange coupling J_ij (scalar, for isotropic Heisenberg term)
EDGE_K: Final[str] = "edge_K"  # Biquadratic exchange coupling K_ij (scalar, for (S_i·S_j)² term)

# =============================================================================
# EDGE ENERGY CONTRIBUTIONS (before reduction to per-atom)
# =============================================================================

EDGE_ENERGY_HEGNN: Final[str] = "edge_energy_HEGNN"  # Heisenberg EGNN edge energy (legacy)
EDGE_ENERGY_SEGNN: Final[str] = "edge_energy_SEGNN"  # Steerable EGNN edge energy (legacy)
EDGE_ENERGY_BQ: Final[str] = "edge_energy_BQ"        # Biquadratic: K_ij * (S_i·S_j)²
EDGE_ENERGY_J: Final[str] = "edge_energy_J"          # Exchange tensor: S_i^T J_ij S_j
EDGE_ENERGY_A: Final[str] = "edge_energy_A"          # Anisotropy contribution per edge
EDGE_ENERGY_TENN: Final[str] = "edge_energy_TENN"    # TENN edge energy contribution    
    
# =============================================================================
# FIELD REGISTRATION FOR NEQUIP DATA HANDLING
# =============================================================================
# Register custom fields so NequIP's data infrastructure can handle them properly
# during batching, transfer to device, etc.

# Node fields: per-atom quantities
register_fields(node_fields=[NODE_SPIN_LENGTH, NODE_SPIN_VEC])

# Edge fields: per-edge quantities (includes all spin and energy edge fields)
register_fields(edge_fields=[
    EDGE_ENERGY, EDGE_FEATURES, EDGE_SPIN,
    EDGE_SPIN_DISTANCE, EDGE_SPIN_DISTANCE_EMBEDDING,
    EDGE_J, EDGE_ENERGY_HEGNN, EDGE_ENERGY_SEGNN,
    EDGE_K, EDGE_ENERGY_BQ,
    EDGE_FEATURES_MSENN_J, EDGE_FEATURES_MSENN_A,
    EDGE_ENERGY_J, EDGE_ENERGY_A,
    EDGE_ENERGY_TENN
])

# Graph fields: per-structure quantities (reduced from per-atom)
register_fields(graph_fields=[
    PER_ATOM_SPIN_KEY,
    PER_ATOM_ENERGY_HEGNN, PER_ATOM_ENERGY_SEGNN,
    PER_ATOM_ENERGY_BQ, PER_ATOM_ENERGY_J, PER_ATOM_ENERGY_A,
    PER_ATOM_ENERGY_TENN
])
>>>>>>> 8da7a7f (fixup! Finished simple tutorials)
