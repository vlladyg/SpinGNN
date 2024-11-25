"""Keys file to overcome TorchScript constants bug."""

import sys

if sys.version_info[1] >= 8:
    from typing import Final
else:
    from typing_extensions import Final

from nequip.data import register_fields

EDGE_ENERGY: Final[str] = "edge_energy"
EDGE_FEATURES: Final[str] = "edge_features"

PER_ATOM_ENERGY_HEGNN: Final[str] = "atomic_energy_HEGNN"
PER_ATOM_ENERGY_SEGNN: Final[str] = "atomic_energy_SEGNN"
PER_ATOM_SPIN_KEY: Final[str] = "atomic_spin"

NODE_SPIN: Final[str] = "node_spin"
NODE_SPIN_LENGTH: Final[str] = "node_spin_length"
EDGE_SPIN: Final[str] = "edge_spin"
EDGE_SPIN_DISTANCE: Final[str] = "edge_spin_distance"
EDGE_SPIN_DISTANCE_EMBEDDING: Final[str] = "edge_spin_distance_embdedding"

EDGE_J: Final[str] = "edge_J"
EDGE_ENERGY_HEGNN: Final[str] = "edge_energy_HEGNN"
EDGE_ENERGY_SEGNN: Final[str] = "edge_energy_SEGNN"

register_fields(node_fields=[NODE_SPIN, NODE_SPIN_LENGTH])
register_fields(edge_fields=[EDGE_ENERGY, EDGE_FEATURES, EDGE_SPIN, 
                             EDGE_SPIN_DISTANCE, EDGE_SPIN_DISTANCE_EMBEDDING, 
                             EDGE_J, EDGE_ENERGY_HEGNN, EDGE_ENERGY_SEGNN])
register_fields(graph_fields=[PER_ATOM_SPIN_KEY, PER_ATOM_ENERGY_HEGNN, PER_ATOM_ENERGY_SEGNN])
