"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

from nequip.data import register_fields

# [n_edge, 1]: define edge atomic type Zij as Zi * num_types + Zj
EDGE_TYPE_KEY: Final[str] = "edge_types"

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
PER_ATOM_SPIN_KEY: Final[str] = "atomic_magmoms"  

    
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
register_fields(graph_fields=[PER_ATOM_SPIN_KEY, PER_ATOM_ENERGY_HEGNN, PER_ATOM_ENERGY_SEGNN,
                              PER_ATOM_ENERGY_BQ, PER_ATOM_ENERGY_J, PER_ATOM_ENERGY_A,
                              PER_ATOM_ENERGY_TENN])
