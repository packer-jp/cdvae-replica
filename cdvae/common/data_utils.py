import numpy as np
import pandas as pd
import networkx as nx
import torch
import copy
import itertools

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from networkx.algorithms.components import is_connected

from sklearn.metrics import accuracy_score, recall_score, precision_score

from torch_scatter import scatter

from p_tqdm import p_umap


# Tensor of unit cells. Assumes 27 cells in -1, 0, 1 offsets in the x and y dimensions
# Note that differing from OCP, we have 27 offsets here because we are in 3D
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

EPSILON = 1e-5

chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og'
]

CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1,
    porous_adjustment=False
)

def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt='cif')

    if primitive:
        crystal = crystal.get_primitive_structure()

    if niggli:
        crystal = crystal.get_reduced_structure()
    
    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    # match is gaurantteed because cif only uses lattice params & frac_coords
    # assert canonical_crystal.matches(crystal)
    return canonical_crystal


def build_crystal_graph(crystal, graph_method="crystalnn"):
    """
    """
    
    if graph_method == "crystalnn":
        crystal_graph = StructureGraph.with_local_env_strategy(crystal, strategy=CrystalNN)
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError
    
    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]
    
    assert np.allclose(
        crystal.lattice.matrix,
        lattice_params_to_matrix(*lengths, *angles),
    )

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([i, j])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))
    
    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atom = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atom


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


class StandardScalerTorch(object):
    """Normalize the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        X = torch.tensor(X, dtype=torch.float)
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return (X - self.means) / self.stds
    
    def inverse_transform(self, X):
        X = torch.tensor(X, dtype=torch.float)
        return X * self.stds + self.means
    
    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(),
            stds=self.stds.clone().detach(),
        )
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means={self.means.tolist()}, "
            f"stds={self.stds.tolist()})"
        )


def get_scaler_from_data_list(data_list, key):
    targets = torch.tensor([d[key]] for d in data_list)
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler


def preprocess(input_file, num_workers, niggli, primitive, graph_method, prop_list):
    df = pd.read_csv(input_file)

    def process_one(row, niggli, primitive, graph_method, prop_list):
        crystal_str = row["cif"]
        crystal = build_crystal(crystal_str, niggli=niggli, primitive=primitive)
        graph_arrays = build_crystal_graph(crystal, graph_method=graph_method)
        properties = {k: row[k] for k in prop_list if k in row.keys()}
        result_dict = {
            "mp_id": row["material_id"],
            "cif": crystal_str,
            "graph_arrays": graph_arrays,
        }
        result_dict.update(properties)
        return result_dict
    
    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))]
        [niggli] * len(df),
        [primitive] * len(df),
        [graph_method] * len(df),
        [prop_list] * len(df),
        num_cpus=num_workers,
    )

    mpid_to_results = {result["mp_id"]: result for result in unordered_results}
    ordered_results = [mpid_to_results[df.iloc[idx]["material_id"]]
                       for idx in range(len(df))]
    
    return ordered_results


def preprocess_tensors(crystal_array_list, niggli, primitive, graph_method):
    def process_one(batch_idx, crystal_array, niggli, primitive, graph_method):
        frac_coords = crystal_array["frac_coords"]
        atom_types = crystal_array["atom_types"]
        lengths = crystal_array["lengths"]
        angles = crystal_array["angles"]
        crystal = Structure(
            lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        graph_arrays = build_crystal_graph(crystal, graph_method)
        result_dict = {
            "batch_idx": batch_idx,
            "graph_arrays": graph_arrays,
        }
        return result_dict
    
    unordered_results = p_umap(
        process_one,
        list(range(len(crystal_array_list))),
        crystal_array_list,
        [niggli] * len(crystal_array_list),
        [primitive] * len(crystal_array_list),
        [graph_method] * len(crystal_array_list),
        num_cpus=30,
    )
    ordered_results = list(sorted(unordered_results, key=lambda x: x["batch_idx"]))
    return ordered_results


def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for dict in data_list:
        graph_arrays = dict["graph_arrays"]
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == "scale_length":
            length = lengths / float(num_atoms) ** (1 / 3)

        dict["scaled_lattice"] = np.concatenate([length, angles])