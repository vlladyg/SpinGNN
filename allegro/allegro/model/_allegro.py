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
    EdgewiseEnergySumSEGNN,
    EdgewiseEnergySumHEGNN,
    EdgewiseSpinSum,
    AtomwiseReduceSpinGNN,
    Allegro_Module,
    Allegro_Module_SEGNN,
    ScalarMLP,
)
#from allegro._keys import EDGE_FEATURES, EDGE_ENERGY, EDGE_SPIN, EDGE_SPIN_DISTANCE_EMBEDDING, EDGE_J
#from allegro._keys import EDGE_ENERGY_SEGNN
from allegro._keys import *
from allegro import RadialBasisSpinDistanceEncoding


from nequip.model import builder_utils


def Allegro(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
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


    print(config)
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
        "allegro_HEGNN": (
            Allegro_Module,
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
        "edge_J": (
            ScalarMLP,
            dict(field=EDGE_FEATURES, out_field=EDGE_J, 
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
        # Sum edgewise exchange terms -> per-atom energies of exchange terms:
        "edge_eng_sum_HEGNN": EdgewiseEnergySumHEGNN,
        # The SEGNN allegro model:
        "allegro_SEGNN": (
            Allegro_Module_SEGNN,
            dict(
                field=AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                edge_invariant_field=AtomicDataDict.EDGE_EMBEDDING_KEY,
                node_invariant_field=AtomicDataDict.NODE_ATTRS_KEY,
            ),
        ),
        "edge_eng_SEGNN": (
            ScalarMLP,
            dict(field=EDGE_FEATURES, out_field=EDGE_ENERGY_SEGNN, 
                 mlp_latent_dimensions = [], mlp_output_dimension=1),
        ),
        "edge_spin": (
            ScalarMLP,
            dict(field=EDGE_FEATURES, out_field=EDGE_SPIN, 
                 mlp_latent_dimensions = [], mlp_output_dimension=1),
        ),
        # Sum SEGNN energy sum
        "edge_eng_sum_SEGNN": EdgewiseEnergySumSEGNN,
        # Sum spins -> per-atom spins
        "edge_eng_spin": EdgewiseSpinSum,
        
        # Sum system energy:
        "total_energy_sum": (
            AtomwiseReduceSpinGNN,
            dict(
                reduce="sum",
                field1=AtomicDataDict.PER_ATOM_ENERGY_KEY,
                field2=PER_ATOM_ENERGY_HEGNN,
                field3=PER_ATOM_ENERGY_SEGNN,
                out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
            ),
        ),
    }
    #print(config["edge_eng"])
    model = SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)

    return model
