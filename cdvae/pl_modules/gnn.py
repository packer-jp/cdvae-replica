"""This module is adapted from https://github.com/Open-Catalyst-Project/ocp/tree/master/ocpmodels/models
"""

import torch
import torch.nn as nn
from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn.models.dimenet import (
    BesselBasisLayer,
    EmbeddingBlock,
    ResidualLayer,
    SphericalBasisLayer,
)
from torch_sparse import SparseTensor

from cdvae.common.data_utils import (
    get_pbc_distances,
    frac_to_cart_coords,
    radius_graph_pbc_wrapper,
)
from cdvae.pl_modules.gemnet.gemnet import GEMNetT

try:
    import sympy as sym
except ImportError:
    sym = None

class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        int_emb_size,
        basis_emb_size, # 格子定数の embedding size
        num_spherical,
        num_before_skip,
        num_after_skip,
        act=swish, # 活性化関数 swish: softReLU 的なもの
    ):
        super(InteractionPPBlock, self).__init__()
        self.act = act