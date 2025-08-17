import sys
sys.path.append('/nfs_beijing_ai/jinxian/rama-scoring1.3.0')
import numpy as np
import os

import torch

import utils.constants.residue_constants as residue_constants
import utils.constants.atom_constants as atom_constants

from utils.logger import Logger

logger = Logger.logger

# coarse graining rigid body
a_res_1st_cg_num = {
    "ALA": 0,
    "ARG": 2,
    "ASP": 6,
    "ASN": 9,
    "CYS": 12,
    "GLU": 15,
    "GLN": 18,
    "GLY": 21,
    "HIS": 23,
    "ILE": 26,
    "LEU": 30,
    "LYS": 33,
    "MET": 37,
    "PHE": 40,
    "PRO": 43,
    "SER": 46,
    "THR": 49,
    "TRP": 52,
    "TYR": 55,
    "VAL": 58,
    "UNK": 61,
}

cg_dict = {
    "ALA": [("C", "CA", "CB", "N"), ("C", "CA", "O")],
    "ARG": [
        ("C", "CA", "CB", "N"),
        ("C", "CA", "O"),
        ("CB", "CG", "CD"),
        ("NE", "NH1", "NH2", "CZ"),
    ],
    "ASP": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "OD1", "OD2")],
    "ASN": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "ND2", "OD1")],
    "CYS": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CA", "CB", "SG")],
    "GLU": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD", "OE1", "OE2")],
    "GLN": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD", "OE1", "NE2")],
    "GLY": [("C", "CA", "N"), ("C", "CA", "O")],
    "HIS": [
        ("C", "CA", "CB", "N"),
        ("C", "CA", "O"),
        ("CG", "CD2", "CE1", "ND1", "NE2"),
    ],
    "ILE": [
        ("C", "CA", "CB", "N"),
        ("C", "CA", "O"),
        ("CB", "CG1", "CG2"),
        ("CB", "CG1", "CD1"),
    ],
    "LEU": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CD1", "CD2")],
    "LYS": [
        ("C", "CA", "CB", "N"),
        ("C", "CA", "O"),
        ("CB", "CG", "CD"),
        ("CD", "CE", "NZ"),
    ],
    "MET": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CG", "CE", "SD")],
    "PHE": [
        ("C", "CA", "CB", "N"),
        ("C", "CA", "O"),
        ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    ],
    "PRO": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CB", "CG", "CD")],
    "SER": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CA", "CB", "OG")],
    "THR": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CB", "CG2", "OG1")],
    "TRP": [
        ("C", "CA", "CB", "N"),
        ("C", "CA", "O"),
        ("CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "NE1"),
    ],
    "TYR": [
        ("C", "CA", "CB", "N"),
        ("C", "CA", "O"),
        ("CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    ],
    "VAL": [("C", "CA", "CB", "N"), ("C", "CA", "O"), ("CB", "CG1", "CG2")],
    "UNK": [()],
}

# max num atoms in CG, N_CG_MAX=9
N_CG_MAX = max([max(len(y) for y in x) for x in cg_dict.values()])

# check for completeness
for res, atoms in residue_constants.residue_atoms.items():
    assert set([atom for cg in cg_dict[res] for atom in cg]) == set(atoms)

# cg:= (residue index, CG index)
cg2id = {}
id = 0
for res3 in residue_constants.restypes_3:
    res1 = residue_constants.restype_3to1.get(res3, None)
    if res1 is None:
        continue
    res_id = residue_constants.res_type12id[res1]
    for i in range(len(cg_dict[res3])):
        cg2id[(res_id, i)] = id
        id += 1

id2cg = {i: cg for cg, i in cg2id.items()}
NUM_CG_TYPES = len(cg2id)  # NUM_CG_TYPES := 61

# cgid2atomidlist, mapping cg to its atom 37-index
cgid2atomidlist = {}  # key = cg index (max=61), value = atom index list (max=37)
res2atomfreq = {}
res2atomweight = {}
cg2atomweight = {}
for res_id, j in cg2id.keys():
    res3 = residue_constants.restype_1to3[residue_constants.res_id2type1[res_id]]
    atoms = cg_dict[res3][j]
    cgid = cg2id[(res_id, j)]
    cgid2atomidlist[cgid] = np.asarray([atom_constants.atom2id[x] for x in atoms])
    if res_id not in res2atomfreq.keys():
        res2atomfreq[res_id] = [0] * atom_constants.atom_type_num

    for x in atoms:
        res2atomfreq[res_id][atom_constants.atom2id[x]] += 1

for res_id in res2atomfreq.keys():
    res1 = residue_constants.res_id2type1[res_id]
    res3 = residue_constants.restype_1to3[res1]
    res2atomweight[res_id] = torch.zeros(
        atom_constants.atom_type_num, dtype=torch.float32
    )
    for i, freq in enumerate(res2atomfreq[res_id]):
        res2atomweight[res_id][i] = 1.0 / freq if freq > 0 else 0

for idx, cg in id2cg.items():
    residx, j = cg
    res1 = residue_constants.res_id2type1[residx]
    res3 = residue_constants.restype_1to3[res1]
    atoms_in_cg = cg_dict[res3][j]
    cg2atomweight[idx] = [
        (
            res3,
            atom,
            residue_constants.residue_atoms[res3].index(atom),
            # cg_atom_weight[res3][atom],
            res2atomfreq[residx][atom_constants.atom2id[atom]],
        )
        for atom in atoms_in_cg
    ]


def static_cg_9to37(static_cg):
    static_cg_37 = list()

    atom_type_num = atom_constants.atom_type_num  # atom_type_num := 37
    for idx in range(static_cg.shape[0]):
        cg = id2cg[idx]
        res_id, k = cg
        res_1 = residue_constants.res_id2type1[res_id]
        res_3 = residue_constants.restype_1to3[res_1]
        atoms = cg_dict[res_3][k]

        # logger.info(static_cg[idx])
        pos = torch.zeros((atom_type_num, 3), dtype=torch.float32)
        for i, atom in enumerate(atoms):
            atomid = atom_constants.atom2id[atom]
            # logger.info([res_1, res_3, cg, atom, atomidx, idx, i])
            pos[atomid] = static_cg[idx][i]

        static_cg_37.append(pos)

    static_cg_37 = torch.stack(static_cg_37, dim=0)

    return static_cg_37


def load_static_cg(cg_path):
    static_cg = np.load(cg_path)["x"].astype(np.float32)
    n, m, d = static_cg.shape
    zero = np.zeros((1, m, d), dtype=np.float32)
    static_cg = np.concatenate([static_cg, zero], axis=0)
    static_cg = torch.FloatTensor(static_cg)
    return static_cg


if __name__ == '__main__':
    s_cg = load_static_cg('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/utils/resources/cg_X0.npz')
    print(s_cg[:4])