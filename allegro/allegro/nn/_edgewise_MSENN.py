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
    
    
class EdgewiseEnergySumJ(GraphModuleMixin, torch.nn.Module):
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
            irreps_out={_keys.EDGE_ENERGY_J: "0e", _keys.PER_ATOM_ENERGY_J: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        
        node_spin = data[_keys.NODE_SPIN]
        
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
            irreps_out={_keys.EDGE_ENERGY_J: "0e", _keys.PER_ATOM_ENERGY_A: "0e"},
        )

        self._factor = None
        if normalize_edge_energy_sum and avg_num_neighbors is not None:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        species = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        
        node_spin= data[_keys.NODE_SPIN]
        
        
        A_mtx = torch.einsum('ik,klm->ilm', data[_keys.EDGE_FEATURES_MSENN_A].squeeze(-2), matrix_terms_A)
        
        A_mtx = scatter(A_mtx, edge_center, dim=0, dim_size=len(species))
        
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            A_mtx = A_mtx * factor

        atom_eng_A = torch.einsum('ia,iab,ib->i', node_spin, A_mtx, node_spin).unsqueeze(-1)
        data[_keys.PER_ATOM_ENERGY_A] = atom_eng_A

        return data