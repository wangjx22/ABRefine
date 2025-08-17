import sys
sys.path.append('/nfs_beijing_ai/jinxian/rama-scoring1.3.0')
import pandas as pd
import os
import random
from scipy.spatial.transform import Rotation as R
from np.protein import from_pdb_string
import numpy as np
import pandas as pd
import torch

"""Protein data type."""
import dataclasses
import io
import re
import collections
from typing import Any, Mapping, Optional, Sequence
import numpy as np
import dataclasses
import io
import re
import collections
from typing import Any, Mapping, Optional, Sequence
import numpy as np
from Bio.PDB import PDBParser
from anarci import anarci
from utils.rmsd import superimpose as si
from utils.protein import get_seq_info
from utils.constants import cg_constants
from utils.logger import Logger
from utils.opt_utils import superimpose_single, masked_differentiable_rmsd
import torch
import math
from np import residue_constants
from scipy.spatial.transform import Rotation as R
from utils.logger import Logger
from utils.constants.atom_constants import *
from utils.constants.residue_constants import *
import np.residue_constants as rc
logger = Logger.logger
import matplotlib.pyplot as plt
FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]
PICO_TO_ANGSTROM = 0.01
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
from Bio.PDB import PDBParser, PDBIO, Atom

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

def get_model_from_str(pdb_str):
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    resolution = structure.header['resolution']
    if resolution is None:
        resolution = 0.0
    models = list(structure.get_models())
    model = models[0]
    return model, resolution, structure


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
        if (chain_id is not None) and (chain.get_id() != chain_id):
            continue
        for res in chain:
            has_CA_flag = False
            for atom in res:
                if atom.name == 'CA':
                    has_CA_flag = True
                    break
            if has_CA_flag:
                # seq.append(residue_constants.restype_3to1.get(res.resname, 'X'))
                # residue with X name is considered to be missing in full_seq_AMR, so we have to skip it here.
                # todo: check if this will cause any problem.
                resname = residue_constants.restype_3to1.get(res.resname, 'X')
                if resname != 'X':
                    seq.append(resname)
                else:
                    continue
    seq = "".join(seq)
    return seq


def get_seq_info(input_seq):
    """Run  get_seq_info method."""
    # code.
    cleaned_seq = ""
    last_is_aa = True
    is_add_list = []
    for aa in input_seq:
        if aa in "[]":
            if aa == "[":
                cur_is_aa = False
        else:
            cleaned_seq += aa
            if last_is_aa:
                is_add_list.append(0)
            else:
                is_add_list.append(1)
            cur_is_aa = True
        last_is_aa = cur_is_aa
        if aa == "X":
            is_add_list.pop()
            cleaned_seq = cleaned_seq[:-1]

    return cleaned_seq, is_add_list



train_file_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset_valid'
# train_file_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/filter_data_from_4_dataset'
df = load_data(train_file_path+'.csv')

# chain_ids = list(df["chain"])
# pdb_files = list(df["pdb_fpath"])
# pdb_files_gt = list(df["pdb_fpath_gt"]) 
# fv_seq_amr = list(df["full_seq_AMR"]) 
# pdb_file_path = '/nfs_beijing_ai/ziqiao-2/tmp/rama-scoring-v1.2.0-decoys/41/8be2_N_decoy_nb15_0.pdb' # start == gt
# pdb_file_path = '/nfs_beijing_ai/ziqiao-2/tmp/rama-scoring-v1.2.0-decoys/41/3zhk_B_decoy_nb22_15.pdb'# start < gt 
# pdb_file_path = '/nfs_beijing_ai/share/sunyiwu/alphaflow_decoys/2p42_D/2p42_D.053.pdb' # start > gt
# position = pdb_files.index(pdb_file_path)
# full_seq_AMR =  fv_seq_amr[position]
# chain_id = chain_ids[position]
# pdb_files_gt =  pdb_files_gt[position]


import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
# df = pd.read_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522.csv')

# 统计列A中独一无二的值和每个对应的B中唯一值的数量
unique_counts = df.groupby("id")['pdb_fpath'].nunique().reset_index()

# 打印独一无二的值和对应的B值数量
print(unique_counts)
# 打印数量的范围
min_count = unique_counts['pdb_fpath'].min()
max_count = unique_counts['pdb_fpath'].max()
print(f"数量范围：最小值 = {min_count}, 最大值 = {max_count}")
max_row = unique_counts[unique_counts['pdb_fpath'] == max_count]
print(max_row)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.bar(unique_counts['id'].astype(str), unique_counts['pdb_fpath'], color='skyblue')
plt.xlabel('Unique Values in Column A')
plt.ylabel('Count of Unique Corresponding B Values')
plt.title('Count of Unique B Values for Each Unique A Value')
plt.xticks(rotation=45)
# 保存图像到指定位置
plt.tight_layout()
plt.savefig('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/plot_filter.png')
plt.show()

print('full_seq_AMR:', full_seq_AMR)
print('chain_ids:', chain_id)
print('pdb_files_gt:', pdb_files_gt)

with open(pdb_file_path, 'r') as f:
    pdb_str = f.read()
        
with open(pdb_files_gt, 'r') as f:
    gt_pdb_str = f.read()


model, resolution, structure = get_model_from_str(pdb_str)
gt_model, _, _structure = get_model_from_str(gt_pdb_str)



# full_seq_AMR is given, need to check the AMR and cut_fv issue.
pdb_seq = get_pdb_seq_by_CA(model, chain_id=None)
print('pdb_seq:',pdb_seq)
cleaned_seq, is_add_list = get_seq_info(full_seq_AMR) # cleaned_seq is full seq include missing res

print('cleaned_seq:',cleaned_seq)
po = from_pdb_string(pdb_str=pdb_str, gt_pdb_str=gt_pdb_str, chain_id=chain_id, ca_only=False, full_seq_AMR=full_seq_AMR, data_type='train')
