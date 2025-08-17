"""Constants used in AlphaFold."""
import os
from typing import Mapping
import pkg_resources

import csv
import pandas as pd
import numpy as np


ele2num = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}
residue_atoms = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'N', 'NE1', 'O'],
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O', 'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O'],
}
residue_atom_renaming_swaps = {
    'ASP': {'OD1': 'OD2'},
    'GLU': {'OE1': 'OE2'},
    'PHE': {'CD1': 'CD2', 'CE1': 'CE2'},
    'TYR': {'CD1': 'CD2', 'CE1': 'CE2'},
}
atom_types = [
    'N',
    'CA',
    'C',
    'CB',
    'O',
    'CG',
    'CG1',
    'CG2',
    'OG',
    'OG1',
    'SG',
    'CD',
    'CD1',
    'CD2',
    'ND1',
    'ND2',
    'OD1',
    'OD2',
    'SD',
    'CE',
    'CE1',
    'CE2',
    'CE3',
    'NE',
    'NE1',
    'NE2',
    'OE1',
    'OE2',
    'CH2',
    'NH1',
    'NH2',
    'OH',
    'CZ',
    'CZ2',
    'CZ3',
    'NZ',
    'OXT',
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)

restypes = [
    'A',
    'R',
    'N',
    'D',
    'C',
    'Q',
    'E',
    'G',
    'H',
    'I',
    'L',
    'K',
    'M',
    'F',
    'P',
    'S',
    'T',
    'W',
    'Y',
    'V',
]

restype_order = {restype: i for i, restype in enumerate(restypes)} # this is type to id
restype_num = len(restypes)
unk_restype_index = restype_num
restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for i, restype in enumerate(restypes_with_x)}
restype_order_with_x_inv = {i: restype for i, restype in enumerate(restypes_with_x)}

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}
restype_3to1 = {v: k for k, v in restype_1to3.items()}

AA_to_tip = {
    'ALA': 'CB',
    'CYS': 'SG',
    'ASP': 'CG',
    'ASN': 'CG',
    'GLU': 'CD',
    'GLN': 'CD',
    'PHE': 'CZ',
    'HIS': 'NE2',
    'ILE': 'CD1',
    'GLY': 'CA',
    'LEU': 'CG',
    'MET': 'SD',
    'ARG': 'CZ',
    'LYS': 'NZ',
    'PRO': 'CG',
    'VAL': 'CB',
    'TYR': 'OH',
    'TRP': 'CH2',
    'SER': 'OG',
    'THR': 'OG1',
}

aas = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
       'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residuemap = dict([(aas[i], i) for i in range(len(aas))])
olt = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
       'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
aanamemap = dict([(aas[i], olt[i]) for i in range(len(aas))])

atypes = {}
types = {}
ntypes = 0
script_dir = os.path.dirname(__file__)
location = pkg_resources.resource_filename(__name__, 'property/aas20.txt')
with open(location, 'r') as f:
    data = csv.reader(f, delimiter=' ')
    for line in data:
        if line[1] in types:
            atypes[line[0]] = types[line[1]]
        else:
            types[line[1]] = ntypes
            atypes[line[0]] = ntypes
            ntypes += 1
location = pkg_resources.resource_filename(__name__, 'property/BLOSUM62.txt')
blosum = [i.strip().split() for i in open(location).readlines()[1:-1]]
blosummap = dict([(l[0], np.array([int(i) for i in l[1:]]) / 10.0) for l in blosum])
location = pkg_resources.resource_filename(__name__, 'property/Meiler.csv')
temp = pd.read_csv(location).values
meiler_features = dict([(t[0], t[1:]) for t in temp])


def sequence_to_onehot(sequence: str, mapping: Mapping[str, int],
                       map_unknown_to_x: bool = False) -> np.ndarray:
    """Run sequence_to_onehot method."""
    # code.
    """Maps the given sequence into a one-hot encoded matrix.

    Args:
      sequence: An amino acid sequence.
      mapping: A dictionary mapping amino acids to integers.
      map_unknown_to_x: If True, any amino acid that is not in the mapping will be
        mapped to the unknown amino acid 'X'. If the mapping doesn't contain
        amino acid 'X', an error will be thrown. If False, any amino acid not in
        the mapping will throw an error.

    Returns:
      A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of
      the sequence.

    Raises:
      ValueError: If the mapping doesn't contain values from 0 to
        num_unique_aas - 1 without any gaps.
    """
    num_entries = max(mapping.values()) + 1
    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError(
            'The mapping must have values from 0 to num_unique_aas-1 without any gaps. Got: %s'
            % sorted(mapping.values()))
    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)
    for aa_index, aa_type in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping['X'])
            else:
                raise ValueError(
                    f'Invalid character in the sequence: {aa_type}')
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1
    return one_hot_arr


def aatype_to_str_sequence(aatype):
    """Run aatype_to_str_sequence method."""
    # code.
    return "".join([restypes_with_x[aatype[i]] for i in range(len(aatype))])
