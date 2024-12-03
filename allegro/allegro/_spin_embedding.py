"""nequip.data.jit: TorchScript functions for dealing with AtomicData.

These TorchScript functions operate on ``Dict[str, torch.Tensor]`` representations
of the ``AtomicData`` class which are produced by ``AtomicData.to_AtomicDataDict()``.

Computing spin distance for nearest neighbors
Authors: Vladimir ladygin
"""

from typing import Dict, Any

import torch
import torch.jit

from typing import Union

from e3nn.util.jit import compile_mode
from e3nn import o3

# Make the keys available in this module
from ._keys import *  # noqa: F403, F401

# Also import the module to use in TorchScript, this is a hack to avoid bug:
# https://github.com/pytorch/pytorch/issues/52312
from . import _keys
from nequip.data import AtomicDataDict
from nequip.nn.radial_basis import BesselBasis

from nequip.nn import GraphModuleMixin


# Define a type alias
Type = Dict[str, torch.Tensor]



@torch.jit.script
def with_edge_spin_length(data: Type, with_distance: bool = True) -> Type:
    """Compute the edge distance vectors between the spins for a graph.

    If ``data.pos.requires_grad`` and/or ``data.cell.requires_grad``, this
    method will return edge vectors correctly connected in the autograd graph.

    Returns:
        Tensor [n_edges, 1] edge distance vectors
        or 
        Tensor [n_nodes, 1] edge nodes if with distance = False
    """
    # Build node spin norms
    if not _keys.NODE_SPIN_LENGTH in data:
        data[_keys.NODE_SPIN_LENGTH] = torch.linalg.norm(data[AtomicDataDict.NODE_SPIN], dim=-1)
        

    # Build spin distance
    if with_distance and _keys.EDGE_SPIN_DISTANCE not in data:
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        unit_spin = data[AtomicDataDict.NODE_SPIN]/data[_keys.NODE_SPIN_LENGTH][:, None]
        data[_keys.NODE_SPIN_VEC] = unit_spin
        
        
        data[_keys.EDGE_SPIN_DISTANCE] = torch.einsum("ki,kj->k", unit_spin[edge_index[1]], unit_spin[edge_index[0]])
        
    return data


@compile_mode("script")
class RadialBasisSpinDistanceEncoding(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        basis_kwargs={},
        out_field: str = _keys.EDGE_SPIN_DISTANCE_EMBEDDING,
        irreps_in=None,
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))]), 
                        _keys.NODE_SPIN_LENGTH: o3.Irreps([(1, (0, 1))])},
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_spin_length(data, with_distance=True)
        edge_spin_distance = data[_keys.EDGE_SPIN_DISTANCE]
        edge_spin_distance_embedded = (
            self.basis(edge_spin_distance) #* self.cutoff(edge_length)[:, None]
        )
        data[self.out_field] = edge_spin_distance_embedded
        return data

    
    
@compile_mode("script")
class SphericalHarmonicEdgeAttrsTENN(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors and nodespins.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh_TENN: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        #out_field: str = 'edge_attr_tmp',
        out_field: str = AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field

        if isinstance(irreps_edge_sh_TENN, int):
            self.irreps_edge_sh_TENN = o3.Irreps.spherical_harmonics(irreps_edge_sh_TENN)
        else:
            self.irreps_edge_sh_TENN = o3.Irreps(irreps_edge_sh_TENN)
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: o3.Irreps([(3, ir) for _, ir in self.irreps_edge_sh_TENN]).sort().irreps},
        )
        self.sh_edge_vec = o3.SphericalHarmonics(
            self.irreps_edge_sh_TENN, edge_sh_normalize, edge_sh_normalization, 
            time_reversal=True, parity = False
        )
        self.sh_node_spin_vec = o3.SphericalHarmonics(
            self.irreps_edge_sh_TENN, edge_sh_normalize, edge_sh_normalization, 
            time_reversal=True, parity = False
        )
        

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        #data = AtomicDataDict.with_edge_vectors(data, with_lengths=False)
        edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY]
        node_spin_vec = data[_keys.NODE_SPIN_VEC]
        
        edge_sh = self.sh_edge_vec(edge_vec).unsqueeze(-2)
        
        edge_node_spin = self.sh_node_spin_vec(node_spin_vec).unsqueeze(-2)
        
        edge_index = data[AtomicDataDict.EDGE_INDEX_KEY]
        edge_node_center = edge_node_spin[edge_index[1]]
        edge_node_neighbor = edge_node_spin[edge_index[0]]
        
        
        # calculating ind for coping (redundant not need)
        #i_shift_0 = 0
        #i_shift_1 = 1
        #i_shift_2 = 2
        #ind_0 = []
        #ind_1 = []
        #ind_2 = []
        #for l in range(self.irreps_edge_sh_TENN.lmax + 1):
        #    ind_0 += list(range(i_shift_0, i_shift_0 + 2*l+1))
        #    ind_1 += list(range(i_shift_1, i_shift_1 + 2*l+1))
        #    ind_2 += list(range(i_shift_2, i_shift_2 + 2*l+1))

        #    i_shift_0 += 3*(2*l+1)
        #    i_shift_1 = i_shift_0 + 2*(l+1) + 1
        #    i_shift_2 = i_shift_1 + 2*(l+1) + 1
        
        #ind_0 = torch.tensor(ind_0, dtype = torch.long)
        #ind_1 = torch.tensor(ind_1, dtype = torch.long)
        #ind_2 = torch.tensor(ind_2, dtype = torch.long)
        
        #data[self.out_field] = torch.zeros((edge_sh.shape[0], self.irreps_out[self.out_field].dim))
        
        #print(edge_sh.shape)
        #print(data[self.out_field].shape)
        #data[self.out_field] = torch.index_copy(data[self.out_field], -1, ind_0, edge_sh)
        #data[self.out_field] = torch.index_copy(data[self.out_field], -1, ind_1, edge_node_center)
        #data[self.out_field] = torch.index_copy(data[self.out_field], -1, ind_2, edge_node_neighbor)
        
        data[self.out_field] = torch.concat([edge_sh, edge_node_center, edge_node_neighbor], dim = -2)
        
        return data