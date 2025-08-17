# atom constant
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom2id = {atom_type: i for i, atom_type in enumerate(atom_types)}
id2atom = {i: atom_type for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.



element2atomic_numbers = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "S": 16,
    "P": 15,
    "F": 9,
    "Cl": 17,
    "Br": 35,
    "I": 53,
    "Se": 34,
    "others": 0,}