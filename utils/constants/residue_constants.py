from importlib import resources
import pickle

from utils.logger import Logger

logger = Logger.logger


restypes_1 = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
]


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}

restype_3to1 = {v: k for k, v in restype_1to3.items()}
restype_3to1.update({"HYP": "P"})
restype_3to1.update({"SEP": "S"})
restype_3to1.update({"PYL": "K"})
restype_3to1.update({"SEC": "C"})
restype_3to1.update({"MSE": "M"})
restype_3to1.update({"DAL": "A"})
restype_3to1.update({"DSN": "S"})
restype_3to1.update({"DCY": "C"})
restype_3to1.update({"DPR": "P"})
restype_3to1.update({"DVA": "V"})
restype_3to1.update({"DTH": "T"})
restype_3to1.update({"DLE": "L"})
restype_3to1.update({"DIL": "I"})
restype_3to1.update({"DSG": "N"})
restype_3to1.update({"DAS": "D"})
restype_3to1.update({"MED": "M"})
restype_3to1.update({"DGN": "Q"})
restype_3to1.update({"DGL": "E"})
restype_3to1.update({"DLY": "K"})
restype_3to1.update({"DHI": "H"})
restype_3to1.update({"DPN": "F"})
restype_3to1.update({"DAR": "R"})
restype_3to1.update({"DTY": "Y"})
restype_3to1.update({"DTR": "W"})
restype_3to1.update({"TPO": "T"})
restype_3to1.update({"YCM": "C"})
restypes_3 = [
    restype_1to3[key] for key in restypes_1
]  # a list of aa type with 3 letters.
res_type12id = {restype: i for i, restype in enumerate(restypes_1)}
res_id2type1 = {idx: restype for restype, idx in res_type12id.items()}
restype_num = len(restypes_1)  # := 21.
unk_restype_index = restype_num  # Catch-all index for unknown restypes.


# A list of atoms (excluding hydrogen) for each AA type. PDB naming convention.
residue_atoms = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": [
        "C",
        "CA",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
        "N",
        "NE1",
        "O",
    ],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
    "UNK": [],
}


# Distance from one CA to next CA [trans configuration: omega = 180].
ca_ca = 3.80209737096

# Between-residue bond lengths for general bonds (first element) and for Proline
# (second element).
between_res_bond_length_c_n = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]

# Between-residue cos_angles.
between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]  # degrees: 121.352 +- 2.315
between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]  # degrees: 116.568 +- 1.995

physis_binary = resources.read_binary("utils.resources", "physics_props_v2.pkl")
physics_props = pickle.loads(physis_binary)


def get_peptide_bond_lengths(resname):
    c_n = (
        between_res_bond_length_c_n[0]
        if resname != "PRO"
        else between_res_bond_length_c_n[1]
    )
    c_n_stddev = (
        between_res_bond_length_stddev_c_n[0]
        if resname != "PRO"
        else between_res_bond_length_stddev_c_n[1]
    )

    return c_n, c_n_stddev


def aatype_to_str_sequence(idx_list):
    return "".join([restypes_1[idx] for idx in idx_list])
