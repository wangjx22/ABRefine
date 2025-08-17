
import dataclasses
import io
import re
import collections
from typing import Any, Mapping, Optional, Sequence
import numpy as np
from np import residue_constants
from Bio.PDB import PDBParser
from anarci import anarci
import pandas as pd
from utils.logger import Logger
from utils.constants.atom_constants import *
from utils.constants.residue_constants import *
from utils.constants.cg_constants import *
from utils.bin_design import *
import np.residue_constants as rc
import math
logger = Logger.logger
def load_data(csv, chain_type=None, filter=None):
    df = pd.read_csv(csv)
    logger.info(f"rows: {len(df)}")
    logger.info(f"filter={filter}")
    if filter == "ca_only":
        for pdb in ab_filter.ca_only_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "backbone_only":
        for pdb in ab_filter.backbone_only_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "long":
        for pdb in ab_filter.long_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "missing_res":
        for pdb in ab_filter.missing_res_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "missing_atom":
        for pdb in ab_filter.missing_atom_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "corner_case":
        for pdb in ab_filter.corner_case_list:
            df = df.drop(df[df["pdb"] == pdb].index) 
    elif filter == "region_le2":
        for pdb in ab_filter.region_le2_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "many_x":
        for pdb in ab_filter.many_x_res_list:
            df = df.drop(df[df["pdb"] == pdb].index)
    elif filter == "all":
        for pdb in ab_filter.ca_only_list:
            df = df.drop(df[df["pdb"] == pdb].index)
        for case in ab_filter.bad_ab_list:
            pdb, chain_id = case.split("-")
            df = df.drop(df[(df["pdb"] == pdb) & (df["chain"] == chain_id)].index)
    else:
        pass
    logger.info(f"rows after drop bad pdbs: {len(df)}")
    if chain_type == "h":
        df = df[df["chain_type"] == chain_type]
        logger.info(f"rows for vh : {len(df)}")
    elif chain_type == "l":
        df = df[(df["chain_type"] == "l") | (df["chain_type"] == "k")]
        logger.info(f"rows for vl : {len(df)}")
    else:
        pass

    return df

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


# pdb = "/pfs_beijing/ai_dataset/abfold_dataset/nb88/pdb/7qe5_B.pdb"
# parser = PDBParser()
# structure = parser.get_structure("X", pdb)
# pdb_seq = get_pdb_seq_by_CA(structure[0], "B")
# print(pdb_seq)

def sample_NB88_data(NB88_csv, NB88_scoring_model_csv):
    NB88_scoring_model = load_data(NB88_scoring_model_csv)
    NB88_scoring_model_pdbid = NB88_scoring_model['id']
    NB88_11 = load_data(NB88_csv)
    NB88_11 = NB88_11[NB88_11['pdb'].isin(list(NB88_scoring_model_pdbid))]
    NB88_11.to_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB88_scoring_model_for_validate.csv', index=False)




NB88_scoring_model_csv = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB88_scoring_model.csv'
NB88_csv = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522.csv'
sample_NB88_data(NB88_csv=NB88_csv, NB88_scoring_model_csv=NB88_scoring_model_csv)