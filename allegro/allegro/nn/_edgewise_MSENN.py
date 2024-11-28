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


class EdgewiseEnergySumTENN(GraphModuleMixin, torch.nn.Module):
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