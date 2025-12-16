"""ETN (Equivariant Tensor Network) Model Builder.

This module provides the model factory for constructing the ETN architecture,
which is fundamentally different from Allegro-based models.

=== ETN vs ALLEGRO/SPINGNN++ ===

| Aspect | Allegro/SpinGNN++ | ETN |
|--------|-------------------|-----|
| Representation | Edge-based | Node-based (from edges) |
| Core operation | Message passing + TP | Tensor train contraction |
| Coupling | Learned weights | Wigner-3j symbols |
| Body order | Fixed by layers | Explicit via depth d |

=== ETN ARCHITECTURE ===

```
Input: positions, atom_types
            |
            v
    +-------------------+
    | Pair Type Embed   |  <-- Encode (type_i, type_j) pairs
    +-------------------+
            |
            v
    +-------------------+
    | Radial Basis      |  <-- Distance encoding
    +-------------------+
            |
            v
    +-------------------+
    | Spherical Harm.   |  <-- Angular encoding
    +-------------------+
            |
            v
    +-------------------+
    | EdgeFeatures_F    |  <-- Combine into F features
    +-------------------+
            |
            v
    +-------------------+
    | EdgewiseFSum      |  <-- Aggregate edges to nodes
    +-------------------+
            |
            v
    +-------------------+
    | ETN_Module        |  <-- Tensor train contraction
    +-------------------+
            |
            v
    +-------------------+
    | AtomwiseReduce    |  <-- Sum to total energy
    +-------------------+
```

=== KEY CONFIGURATION ===

ETN-specific parameters:
- Nc: Number of feature channels
- d: Tensor train depth (body order)
- N_rank_ett: List of TT ranks
- N_rank_spec: Rank for species embedding

Authors: Vladimir Ladygin
"""

from typing import Optional
import logging

from e3nn import o3

from nequip.data import AtomicDataDict, AtomicDataset

from nequip.nn import SequentialGraphNetwork, AtomwiseReduce
from nequip.nn.radial_basis import BesselBasis

from nequip.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
    RadialBasisEdgeEncoding,
)

from allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    EdgewiseEnergySumBQ,
    EdgewiseEnergySumJ,
    EdgewiseEnergySumA,
    EdgewiseEnergySumTENN,
    EdgewiseSpinSum,
    AtomwiseReduceSpinGNNPlus,
    Allegro_Module,
    Allegro_Module_MSENN,
    Allegro_Module_TENN,
    ScalarMLP,
    EdgewiseFSum,
    EdgeFeatures_F,
    ETN_Module
)
from allegro._keys import *
from allegro import PairTypeEmbedding
from allegro import RadialBasisSpinDistanceEncoding, SphericalHarmonicEdgeAttrsTENN


from nequip.model import builder_utils


def ETN(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    """Build the ETN model from configuration.
    
    This factory function constructs the ETN model with the following pipeline:
    
    1. pair_type_embedding: Encode atom type pairs
    2. one_hot: Atom type one-hot encoding
    3. radial_basis: Distance radial basis encoding
    4. spharm: Spherical harmonics of edge vectors
    5. edge_features_F: Construct F features (radial × species × angular)
    6. edge_f_sum: Sum edge features to nodes
    7. etn: Apply tensor train contraction
    8. total_energy_sum: Reduce to total energy
    
    Args:
        config: Configuration dictionary with:
            - Nc: Number of feature channels
            - d: Tensor train depth
            - N_rank_ett: TT rank list
            - N_rank_spec: Species embedding rank
            - l_max: Maximum angular momentum
        initialize: Whether to compute dataset statistics
        dataset: Optional dataset for statistics
    
    Returns:
        SequentialGraphNetwork: The complete ETN model
    """
    logging.debug("Building ETN model...")

    # Handle avg num neighbors auto
    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    # Handle simple irreps
    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config["parity"]
        assert parity_setting in ("o3_full", "o3_restricted", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=-1
            )
        )
        nonscalars_include_parity = parity_setting == "o3_full"
        # check consistant
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        assert (
            config.get("nonscalars_include_parity", nonscalars_include_parity)
            == nonscalars_include_parity
        )
        config["irreps_edge_sh"] = irreps_edge_sh
        config["nonscalars_include_parity"] = nonscalars_include_parity


    #print(config)
    layers = {
        # -- Encode --
        # Get various edge invariants
        "pair_type_embedding": PairTypeEmbedding,
        "one_hot": OneHotAtomEncoding,
        "radial_basis": (
            RadialBasisEdgeEncoding,
            dict(
                basis=(
                    NormalizedBasis
                    if config.get("normalize_basis", True)
                    else BesselBasis
                ),
                out_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            ),
        ),
        # Get edge nonscalars
        "spharm": SphericalHarmonicEdgeAttrs,
        # Get F from ETN paper
        "edge_features_F": (
            EdgeFeatures_F,
            dict(
                Nc = config['Nc'],
                N_rank_spec = config['N_rank_spec'],
                out_field = EDGE_FEATURES_F,
            ),
        ),
        # Sum edgewise f features -> node f features:
        "edge_f_sum": EdgewiseFSum,
        # ETN Layer:
        "etn": (
            ETN_Module,
            dict(
            d = config['d'],
            N_rank_ett = config['N_rank_ett'],
            out_field = AtomicDataDict.PER_ATOM_ENERGY_KEY),
        ),
        # Sum system energy:
        "total_energy_sum": (
            AtomwiseReduce,
            dict(
                reduce="sum",
                field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        ),
    }
    """
    # The HEGNN allegro model:
    "allegro_MSENN": (
        Allegro_Module_MSENN,
        dict(
            field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
            edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
        ),
    ),
    "edge_eng": (
        ScalarMLP,
        dict(field=EDGE_FEATURES, out_field=EDGE_ENERGY, mlp_output_dimension=1),
    ),
    "edge_K": (
        ScalarMLP,
        dict(field=EDGE_FEATURES, out_field=EDGE_K, 
             mlp_latent_dimensions = [], mlp_output_dimension=1),
    ),
    # Sum edgewise energies -> per-atom energies:
    "edge_eng_sum": EdgewiseEnergySum,
    # encoding spin distance
    "spin_basis": (
        RadialBasisSpinDistanceEncoding,
        dict(
            basis=(
                NormalizedBasis
                if config.get("normalize_basis", True)
                else BesselBasis
            ),
            out_field=EDGE_SPIN_DISTANCE_EMBEDDING,
        ),
    ),
    # Sum biquadratic edgewise terms -> per-atom energies of biquadratic terms:
    "edge_eng_sum_BQ": EdgewiseEnergySumBQ,
    # Sum edgewise exchange terms -> per-atom energies of exchange terms:
    "edge_eng_sum_J": EdgewiseEnergySumJ,
    # Sum onsite spin terms -> per-atom energies of onsite spin terms:
    "edge_eng_sum_A": EdgewiseEnergySumA,
    # Get edge nonscalars
    "spharm_TENN": SphericalHarmonicEdgeAttrsTENN,
    # The TENN allegro model:
    "allegro_TENN": (
        Allegro_Module_TENN,
        dict(
            field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
            edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
            node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
        ),
    ),
    "edge_eng_TENN": (
        ScalarMLP,
        dict(field=EDGE_FEATURES, out_field=EDGE_ENERGY_TENN, 
             mlp_latent_dimensions = [], mlp_output_dimension=1),
    ),
    "edge_spin": (
        ScalarMLP,
        dict(field=EDGE_FEATURES, out_field=EDGE_SPIN, 
             mlp_latent_dimensions = [], mlp_output_dimension=1),
    ),
    # Sum SEGNN energy sum
    "edge_eng_sum_TENN": EdgewiseEnergySumTENN,
    # Sum spins -> per-atom spins
    "edge_spin_sum": EdgewiseSpinSum,

    # Sum system energy:
    "total_energy_sum": (
        AtomwiseReduceSpinGNNPlus,
        dict(
            reduce="sum",
            field_eng=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            field_BQ=PER_ATOM_ENERGY_BQ,
            field_J=PER_ATOM_ENERGY_J,
            field_A=PER_ATOM_ENERGY_A,
            field_TENN=PER_ATOM_ENERGY_TENN,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    ),
    """
    #print(config["edge_eng"])
    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model
