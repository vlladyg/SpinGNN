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


class EdgewiseEnergySumBQ(GraphModuleMixin, torch.nn.Module):
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