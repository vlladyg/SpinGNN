from typing import Optional
import math

import torch
from torch_runstats.scatter import scatter

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from .. import _keys


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


class EdgewiseEnergySum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
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
        """Sum edges into nodes."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={_keys.EDGE_ENERGY: "0e"},
            irreps_out={AtomicDataDict.PER_ATOM_ENERGY_KEY: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        self.per_edge_species_scale = per_edge_species_scale
        if self.per_edge_species_scale:
            self.per_edge_scales = torch.nn.Parameter(torch.ones(num_types, num_types))
        else:
            self.register_buffer("per_edge_scales", torch.Tensor())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_eng = data[_keys.EDGE_ENERGY]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        center_species = species[edge_center]
        neighbor_species = species[edge_neighbor]

        if self.per_edge_species_scale:
            edge_eng = edge_eng * self.per_edge_scales[
                center_species, neighbor_species
            ].unsqueeze(-1)

        atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[AtomicDataDict.PER_ATOM_ENERGY_KEY] = atom_eng

        return data


class EdgewiseEnergySumSEGNN(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
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
        """Sum edges into nodes."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={_keys.EDGE_ENERGY_SEGNN: "0e"},
            irreps_out={_keys.PER_ATOM_ENERGY_SEGNN: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        self.per_edge_species_scale = per_edge_species_scale
        if self.per_edge_species_scale:
            self.per_edge_scales_SEGNN = torch.nn.Parameter(torch.ones(num_types, num_types))
        else:
            self.register_buffer("per_edge_scales_SEGNN", torch.Tensor())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_eng = data[_keys.EDGE_ENERGY_SEGNN]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        center_species = species[edge_center]
        neighbor_species = species[edge_neighbor]

        if self.per_edge_species_scale:
            edge_eng = edge_eng * self.per_edge_scales_SEGNN[
                center_species, neighbor_species
            ].unsqueeze(-1)

        atom_eng = scatter(edge_eng, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[_keys.PER_ATOM_ENERGY_SEGNN] = atom_eng

        return data


class EdgewiseEnergySumHEGNN(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
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
        """Sum edges into nodes."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={_keys.EDGE_ENERGY_HEGNN: "0e", _keys.PER_ATOM_ENERGY_HEGNN: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        
        edge_eng_HEGNN = data[_keys.EDGE_J] * data[_keys.EDGE_SPIN_DISTANCE].unsqueeze(-1)
        data[_keys.EDGE_ENERGY_HEGNN] = edge_eng_HEGNN
        
        atom_eng = scatter(edge_eng_HEGNN, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        data[_keys.PER_ATOM_ENERGY_HEGNN] = atom_eng

        return data

class AtomwiseReduceSpinGNN(GraphModuleMixin, torch.nn.Module):
    constant: float

    def __init__(
        self,
        field1: str,
        field2: str,
        field3: str,
        out_field: Optional[str] = None,
        reduce="sum",
        avg_num_atoms=None,
        irreps_in={},
        per_contrib_scales: bool = True,
    ):
        super().__init__()
        assert reduce in ("sum", "mean", "normalized_sum")
        self.constant = 1.0
        if reduce == "normalized_sum":
            assert avg_num_atoms is not None
            self.constant = float(avg_num_atoms) ** -0.5
            reduce = "sum"
        self.reduce = reduce
        self.field1 = field1
        self.field2 = field2
        self.field3 = field3
        self.out_field = f"{reduce}_{field1}" if out_field is None else out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: irreps_in[self.field1]}
            if self.field1 in irreps_in
            else {},
        )

        self.per_contrib_scales = per_contrib_scales
        if self.per_contrib_scales:
            self.per_contrib_scales_SpinGNN = torch.nn.Parameter(torch.ones(3))
        else:
            self.register_buffer("per_contrib_scales_SpinGNN", torch.Tensor())
    

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = AtomicDataDict.with_batch(data)

        term1 = scatter(
            data[self.field1], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )
        term2 = scatter(
            data[self.field2], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )
        term3 = scatter(
            data[self.field3], data[AtomicDataDict.BATCH_KEY], dim=0, reduce=self.reduce
        )

        if self.per_contrib_scales:
            for i, el in enumerate([term1, term2, term3]):
                el *= self.per_contrib_scales_SpinGNN[i]
        
        data[self.out_field] = term1 + term2 + term3
        
        if self.constant != 1.0:
            data[self.out_field] = data[self.out_field] * self.constant
        return data


class EdgewiseSpinSum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """

    _factor: Optional[float]

    def __init__(
        self,
        num_types: int,
        avg_num_neighbors: Optional[float] = None,
        normalize_edge_spin_sum: bool = True,
        per_edge_species_scale: bool = False,
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={_keys.EDGE_SPIN: "0e"},
            irreps_out={_keys.PER_ATOM_SPIN_KEY: "0e"},
        )

        self._factor = None
        if normalize_edge_spin_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        self.per_edge_species_scale = per_edge_species_scale
        if self.per_edge_species_scale:
            self.per_edge_scales_spin = torch.nn.Parameter(torch.ones(num_types, num_types))
        else:
            self.register_buffer("per_edge_scales_spin", torch.Tensor())

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        edge_spin = data[_keys.EDGE_SPIN]
        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        center_species = species[edge_center]
        neighbor_species = species[edge_neighbor]

        if self.per_edge_species_scale:
            edge_spin = edge_spin * self.per_edge_scales_spin[
                center_species, neighbor_species
            ].unsqueeze(-1)

        atom_spin = scatter(edge_spin, edge_center, dim=0, dim_size=len(species))
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_spin = atom_spin * factor

        data[_keys.PER_ATOM_SPIN_KEY] = atom_spin

        return data