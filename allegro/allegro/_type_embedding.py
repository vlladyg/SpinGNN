import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from . import _keys

@compile_mode("script")
class PairTypeEmbedding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {_keys.EDGE_TYPE_KEY: Irreps([(1, (0, 1))])}
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
   

        # Creating flattened square matrix of pair types
        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]
        
        edge_type_embed = (type_numbers[edge_center]*self.num_types + type_numbers[edge_neighbor]).unsqueeze(-1)
        data[_keys.EDGE_TYPE_KEY] = edge_type_embed.to(device=type_numbers.device,
                                                   dtype=data[AtomicDataDict.ATOM_TYPE_KEY].dtype)
        
        return data
