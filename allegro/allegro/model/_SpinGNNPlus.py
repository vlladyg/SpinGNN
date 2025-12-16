"""SpinGNN++ Model Builder.

This module provides the model factory function for constructing the full
SpinGNN++ architecture from a configuration dictionary.

=== SPINGNN++ ARCHITECTURE ===

SpinGNN++ is a machine learning interatomic potential for magnetic materials
that combines:

1. Position-dependent terms (MSENN branch)
2. Spin-dependent terms (TENN branch)
3. Explicit Hamiltonian terms (J, A, K tensors)

The total energy is decomposed as:

    E = E_pair + E_BQ + E_J + E_A + E_TENN

=== MODEL PIPELINE ===

```
Input: positions, atom_types, spins
            |
            v
    +---------------+
    | One-hot atoms |  <-- Atom type encoding
    +---------------+
            |
            v
    +------------------+
    | Radial basis     |  <-- Distance encoding
    | (positions)      |
    +------------------+
            |
            v
    +------------------+
    | Spherical harm.  |  <-- Edge direction encoding
    +------------------+
            |
            v
    +------------------+
    | Allegro_MSENN    |  <-- Position-only Allegro (outputs J, A features)
    +------------------+
            |
    +-------+-------+-------+-------+
    |       |       |       |       |
    v       v       v       v       v
  E_pair  E_BQ    E_J     E_A   (latent)
    |       |       |       |       |
    |       |       |       |       v
    |       |       |       |  +------------------+
    |       |       |       |  | Spin distance    |  <-- S_i · S_j encoding
    |       |       |       |  | encoding         |
    |       |       |       |  +------------------+
    |       |       |       |       |
    |       |       |       |       v
    |       |       |       |  +------------------+
    |       |       |       |  | Sph. harm. TENN  |  <-- Position + spin SH
    |       |       |       |  +------------------+
    |       |       |       |       |
    |       |       |       |       v
    |       |       |       |  +------------------+
    |       |       |       |  | Allegro_TENN     |  <-- Spin-aware Allegro
    |       |       |       |  +------------------+
    |       |       |       |       |
    |       |       |       |       v
    |       |       |       |    E_TENN
    |       |       |       |       |
    +-------+-------+-------+-------+
                    |
                    v
            +---------------+
            | Sum energies  |  <-- AtomwiseReduceSpinGNNPlus
            +---------------+
                    |
                    v
              Total Energy
```

=== CONFIGURATION ===

Key config parameters:
- l_max: Maximum angular momentum for spherical harmonics
- parity: "o3_full", "o3_restricted", or "so3"
- num_layers: Depth of Allegro networks
- env_embed_multiplicity: Width of environment embeddings
- num_basis: Number of radial basis functions

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
)
from allegro._keys import *
from allegro import RadialBasisSpinDistanceEncoding, SphericalHarmonicEdgeAttrsTENN


from nequip.model import builder_utils


def SpinGNNPlus(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    """Build the SpinGNN++ model from configuration.
    
    This factory function constructs the full SpinGNN++ model as a
    SequentialGraphNetwork with the following layers:
    
    1. Encoding layers:
       - one_hot: Atom type one-hot encoding
       - radial_basis: Distance radial basis encoding
       - spharm: Spherical harmonics of edge vectors
    
    2. MSENN branch:
       - allegro_MSENN: Position-only Allegro outputting J/A features
       - edge_eng: MLP for pair energy
       - edge_K: MLP for biquadratic coupling K
       - edge_eng_sum: Sum pair energies to per-atom
    
    3. Spin encoding:
       - spin_basis: Radial basis of spin dot products
       - edge_eng_sum_BQ: Compute biquadratic energy
       - edge_eng_sum_J: Compute exchange tensor energy
       - edge_eng_sum_A: Compute anisotropy energy
    
    4. TENN branch:
       - spharm_TENN: Spherical harmonics of positions AND spins
       - allegro_TENN: Spin-aware Allegro
       - edge_eng_TENN: MLP for TENN energy
       - edge_spin: MLP for per-edge spin contribution
       - edge_eng_sum_TENN: Sum TENN energies
       - edge_spin_sum: Sum spin contributions
    
    5. Final reduction:
       - total_energy_sum: Combine all energy contributions
    
    Args:
        config: Configuration dictionary with model hyperparameters
        initialize: Whether to compute dataset statistics for initialization
        dataset: Optional dataset for computing statistics
    
    Returns:
        SequentialGraphNetwork: The complete SpinGNN++ model
    """
    logging.debug("Building Allegro model...")

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
                l_max, p=(1 if parity_setting == "so3" else -1)
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

    # Handle simple irreps
    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config["parity"]
        assert parity_setting in ("o3_full", "o3_restricted", "so3")
        irreps_edge_sh_TENN = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=1, t = -1
            ) 
        )
        nonscalars_include_parity = parity_setting == "o3_full"
        # check consistant
        config["irreps_edge_sh_TENN"] =  irreps_edge_sh_TENN
        assert config.get("irreps_edge_sh_TENN", irreps_edge_sh_TENN) == irreps_edge_sh_TENN
        assert (
            config.get("nonscalars_include_parity", nonscalars_include_parity)
            == nonscalars_include_parity
        )


    #print(config)
    layers = {
        # -- Encode --
        # Get various edge invariants
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
    }
    #print(config["edge_eng"])
    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model
