
import dataclasses
import io
import re
import collections
from typing import Any, Mapping, Optional, Sequence
import numpy as np
from np import residue_constants
from Bio.PDB import PDBParser
from anarci import anarci

def get_pdb_seq_by_CA(model, chain_id):
    """Run get_pdb_seq_by_CA method."""
    # code.
    """
    get_pdb_seq_by_CA:
    Args:
        model : model
    Returns:
    """
    seq = []
    for chain in model:
        if chain.get_id() != chain_id:
            continue
        for res in chain:
            has_CA_flag = False
            for atom in res:
                if atom.name == 'CA':
                    has_CA_flag = True
                    break
            if has_CA_flag:
                seq.append(residue_constants.restype_3to1[res.resname])
    seq = "".join(seq)
    return seq


pdb = "/pfs_beijing/ai_dataset/abfold_dataset/nb88/pdb/7qe5_B.pdb"
parser = PDBParser()
structure = parser.get_structure("X", pdb)
pdb_seq = get_pdb_seq_by_CA(structure[0], "B")
print(pdb_seq)