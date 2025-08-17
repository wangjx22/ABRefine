import sys
sys.path.append('/nfs_beijing_ai/jinxian/rama-scoring1.3.0')
import argparse
import os
import random
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import utils.mpu_utils as mpu
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch.optim as optim

from np.protein import from_pdb_string, update_pdb_with_new_coords

from dataset.feature.featurizer import process
from utils.data_encoding import encode_structure, encode_features, extract_topology
from dataset.screener import UpperboundScreener, ProbabilityScreener1d
from utils.logger import Logger
from utils.constants.atom_constants import *
from utils.constants.residue_constants import *
import np.residue_constants as rc

logger = Logger.logger
"""Protein data type."""
import dataclasses
import io
import re
import collections
from typing import Any, Mapping, Optional, Sequence
import numpy as np
from np import residue_constants
from Bio.PDB import PDBParser
from anarci import anarci
from utils.opt_utils import superimpose_single
from utils.logger import Logger
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from utils import dist_utils

logger = Logger.logger

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]
PICO_TO_ANGSTROM = 0.01
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)

from model.equiformer_v2_model import EquiformerV2

from torch_geometric.data import Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
torch.autograd.set_detect_anomaly(True)
loss_mse = torch.nn.MSELoss(reduction='mean')
import wandb
import logging
from datetime import datetime

os.environ["WANDB_MODE"] = "run"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None:
        assert seed > 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True  # False
        torch.backends.cuda.matmul.allow_tf32 = (
            False  # if set it to True will be much faster but not accurate
        )
        # if (
        #     not deepspeed_version_ge_0_10 and deepspeed.checkpointing.is_configured()
        # ):  # This version is a only a rough number
        #     mpu.model_parallel_cuda_manual_seed(seed)

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
 

def get_dataset_filepath(csv, sample_num = -1, data_type='train'):
    if sample_num == -1:
        df = load_data(csv)
    else:
        df = load_data(csv).head(sample_num)
  
    # pdb_ids = [list(df["pdb"]) for df in df_list]
    chain_ids = list(df["chain"])
    pdb_files = list(df["pdb_fpath"])
    pdb_files_gt = list(df["pdb_fpath_gt"]) 
    fv_seq_amr = list(df["full_seq_AMR"]) 
    
    data_list = []
    total_data = 0
    for af_pdb, gt_pdb, chain, gt_seq in zip(pdb_files,pdb_files_gt,chain_ids,fv_seq_amr):
        if pd.isna(chain):
            chain = None
        if os.path.exists(af_pdb) and os.path.exists(gt_pdb):
            tem =prepare_features(af_pdb_filepath=af_pdb, gt_pdb_filepath=gt_pdb, chain=chain, full_seq_AMR=gt_seq, data_type=data_type) 
            if tem:
                total_data += 1
                print('total data:', total_data)
                data_list.append(tem)
            else:
                continue
        else:
            continue
        
    return data_list, total_data


def get_cdr_backbone_mask(cdr1_mask_by_res_mask, finally_atom_mask_reshape_r_37):
    """Run the method.
    cdr1_mask_by_res_mask: the cdr res mask and shape is [N], N is the res num
    finally_atom_mask_reshape_r_37: backbone mask, shape is [N, 37]

    output: a cdr backbone atom mask [backbone_num], where index==1 is the atom exist in cdr
    """
    # 寻找第一个和最后一个值为1的索引
    cdr1_begin_index = None
    cdr1_end_index = None
    
    for index in range(len(cdr1_mask_by_res_mask)):
        if cdr1_mask_by_res_mask[index] == 1:
            if cdr1_begin_index is None:
                cdr1_begin_index = index
            cdr1_end_index = index

    cdr1_before_backbone_atome_num  =  finally_atom_mask_reshape_r_37[:cdr1_begin_index, :].sum().item() # cdr1区域之前的backbone原子数量
    cdr1_backbone_atome_num = finally_atom_mask_reshape_r_37[cdr1_begin_index:cdr1_end_index+1, :].sum().item() #cdr1区域的backbone原子数量
    finally_atom_mask_reshape_r_37_backbone_num =  finally_atom_mask_reshape_r_37.sum().item()
    cdr1_mask_backbone_atom = [0] * finally_atom_mask_reshape_r_37_backbone_num #backbone all atom
    cdr1_mask_backbone_atom[cdr1_before_backbone_atome_num: cdr1_before_backbone_atome_num + cdr1_backbone_atome_num] = [1] * cdr1_backbone_atome_num
    
    return cdr1_mask_backbone_atom


def prepare_features(gt_pdb_filepath, af_pdb_filepath, chain, full_seq_AMR, data_type='train', dataset_type=None):
    """Run prepare_features method.
    gt_pdb_filepath: a file path of gt_pdb_filepath
    af_pdb_filepath: a file path of af_pdb_filepath
    chain: the chain of pdb

    output: a torch_geometric data of the pdb
    """
    
    # code.
    
    with open(af_pdb_filepath, 'r') as f:
        pdb_str = f.read()
    
    if data_type =='train':
        with open(gt_pdb_filepath, 'r') as f:
            gt_pdb_str = f.read()

        try:
            # print("Now is: ", af_pdb_filepath, "and ", gt_pdb_filepath)
            po = from_pdb_string(pdb_str=pdb_str, gt_pdb_str=gt_pdb_str, chain_id=chain, ca_only=False, full_seq_AMR=full_seq_AMR, data_type=data_type, dataset_type=dataset_type)
        except ValueError as e: 
            print("Errors: ", af_pdb_filepath, "and ", gt_pdb_filepath)
        
        if po == False:
            return False

        # extend element informations
        
        gt_atom_mask = torch.from_numpy(eval(f"po.{'gt_atom_mask'}").astype(np.float32))
        
        atom2cgids = torch.from_numpy(eval(f"po.{'atom2cgids'}").astype(np.float32))
        gt_atom_positions = torch.from_numpy(eval(f"po.{'gt_atom_positions'}").astype(np.float32))
        
        all_atom_positions = torch.from_numpy(eval(f"po.{'atom_positions'}").astype(np.float32))
        
        rmsd = torch.from_numpy(eval(f"po.{'rmsd'}").astype(np.float32))

        trans_target = torch.from_numpy(eval(f"po.{'trans_target'}").astype(np.float32))
        rot_target = torch.from_numpy(eval(f"po.{'rot_target'}").astype(np.float32))
        rot_m4_target = torch.from_numpy(eval(f"po.{'rot_m4_target'}").astype(np.float32))

        pdb_name = af_pdb_filepath

        gt_res_mask = torch.from_numpy(eval(f"po.{'gt_res_mask'}"))
        if all_atom_positions.shape[0] != gt_res_mask.shape[0]:
            return False

        po2struc = {
            'atom_positions': 'xyz',
            'atom_names': 'name',
            'elements': 'element',
            'residue_names': 'resname',
        }
        structure = {}
        
        #find the backbone atom, where the index is 0,1,2,4
        res_list = torch.zeros(37)
        res_list[0], res_list[1], res_list[2], res_list[4] = 1,1,1,1

        # align data
        align_atom_positions = torch.from_numpy(eval(f"po.{'align_pos'}").astype(np.float32))
        # true_align_atom_positions = align_atom_positions.view(-1,3)[gt_atom_mask.view(-1)==1]

        start_atom_mask = po.atom_mask
        
        atom_mask = start_atom_mask[gt_res_mask==1].reshape(-1).astype(bool)
        atom_mask = torch.from_numpy(atom_mask)
        start_atom_mask = torch.from_numpy(start_atom_mask)

        # 获得初始化的backbone list
        gt_res_num = align_atom_positions.shape[0]
        backbone_list = res_list.repeat(gt_res_num, 1).view(-1)
        
        # 获取从atom mask中继续获得backbone原子的位置，其中1代表backbone原子
        finally_atom_mask = torch.logical_and(atom_mask, backbone_list).int()
        # align_atom_mask_tran = align_atom_positions.view(-1,3)[finally_atom_mask==1]
        # X = align_atom_mask_tran
        atom_trans_align2gt = torch.from_numpy(eval(f"po.{'atom_trans_align2gt'}").astype(np.float32))
        align_atom_trans = atom_trans_align2gt.view(-1,3)[finally_atom_mask==1]
        

        # judge the dataset_type
        if dataset_type == 'NB':
            cdr1_mask = torch.from_numpy(eval(f"po.{'cdr1_mask'}"))
            cdr2_mask = torch.from_numpy(eval(f"po.{'cdr2_mask'}"))
            cdr3_mask = torch.from_numpy(eval(f"po.{'cdr3_mask'}"))

            # frist select the mask from res_mask
            cdr1_mask_by_res_mask = cdr1_mask[gt_res_mask==1]
            cdr2_mask_by_res_mask = cdr2_mask[gt_res_mask==1]
            cdr3_mask_by_res_mask = cdr3_mask[gt_res_mask==1]

            # select the atom mask and backbone_list mask
            finally_atom_mask_reshape_r_37 = finally_atom_mask.reshape((-1,37))
            # set the pos =0 of backbone_reshape_r_37 if the atom not in cdr 
            # 只要计算cdr前面的和自身区域1数量即可
            cdr1_mask_backbone_atom = torch.tensor(get_cdr_backbone_mask(cdr1_mask_by_res_mask, finally_atom_mask_reshape_r_37))
            cdr2_mask_backbone_atom = torch.tensor(get_cdr_backbone_mask(cdr2_mask_by_res_mask, finally_atom_mask_reshape_r_37))
            cdr3_mask_backbone_atom = torch.tensor(get_cdr_backbone_mask(cdr3_mask_by_res_mask, finally_atom_mask_reshape_r_37))

            # cdr1_begin_index = cdr1_mask_by_res_mask.index(1) # 是以res为单位
            # cdr1_end_index = len(cdr1_mask_by_res_mask) - 1 - cdr1_mask_by_res_mask[::-1].index(1)
            
            # cdr1_before_backbone_atome_num  =  finally_atom_mask_reshape_r_37[:cdr1_begin_index, :].sum().item() # cdr1区域之前的backbone原子数量
            # cdr1_backbone_atome_num = finally_atom_mask_reshape_r_37[cdr1_begin_index:cdr1_end_index+1, :].sum().item() #cdr1区域的backbone原子数量
            
            # backbone_atom_num_list = [0] * finally_atom_mask_reshape_r_37.sum().item() #backbone all atom
            # cdr1_mask_backbone_atom[cdr1_before_backbone_atome_num: cdr1_before_backbone_atome_num + cdr1_backbone_atome_num+1] = [1] * cdr1_backbone_atome_num

        else:
            finally_atom_mask_backbone_num =  finally_atom_mask.sum().item()
            cdr1_mask_backbone_atom = [0] * finally_atom_mask_backbone_num #backbone all atom
            cdr1_mask_backbone_atom = torch.tensor(cdr1_mask_backbone_atom)
            cdr2_mask_backbone_atom = torch.tensor(cdr1_mask_backbone_atom)
            cdr3_mask_backbone_atom = torch.tensor(cdr1_mask_backbone_atom)

        for k in [
        # 'atom_positions',
        'atom_names',
        'residue_names',
        'elements',
        ]:
            item = eval(f"po.{k}")
            item = item[gt_res_mask==1].reshape(-1, )
            item = item[finally_atom_mask==1] # mask the 0
            structure[po2struc[k]] = item

        atom_names = structure['name'].tolist()
        atom_numbers = torch.Tensor([atom2id[atname] for atname in atom_names])     # atom number of atoms in residues. max number is 37.

        elements = structure['element'].tolist()
        atomic_numbers = [element2atomic_numbers[element]for element in elements]
        atomic_numbers = torch.Tensor(atomic_numbers)

        res_names = structure['resname'].tolist()
        res1 = [restype_3to1[resname] for resname in res_names]
        resid = [res_type12id[res1[i]] for i in range(len(res1))]
        resid = torch.Tensor(resid)
        gt_backbone_atom_positions = gt_atom_positions.view(-1,3)[finally_atom_mask==1]
        X = align_atom_positions.view(-1,3)[finally_atom_mask==1]
        n_nodes = X.size(0)


        data = Data(
            pos=X,
            atom_numbers=atom_numbers, 
            atomic_numbers=atomic_numbers, 
            start_atom_mask=start_atom_mask,
            resid=resid, 
            n_nodes=n_nodes,
            gt_atom_positions=gt_atom_positions,
            gt_backbone_atom_positions = gt_backbone_atom_positions,
            gt_atom_mask=gt_atom_mask,
            gt_res_mask=gt_res_mask,
            cdr1_mask_backbone_atom = cdr1_mask_backbone_atom,
            cdr2_mask_backbone_atom = cdr2_mask_backbone_atom,
            cdr3_mask_backbone_atom = cdr3_mask_backbone_atom,
            all_atom_positions=all_atom_positions, # start model pos
            align_atom_positions = align_atom_positions,
            align_atom_trans = align_atom_trans,
            atom2cgids= atom2cgids,
            pdb_name = pdb_name,
            trans_target= trans_target,
            rot_target= rot_target,
            rot_m4_target = rot_m4_target,
            rmsd = rmsd)

    elif data_type =='test': # 注意X应该是将start model的全部backbone都输入，而不是输入align model
        with open(gt_pdb_filepath, 'r') as f:
            gt_pdb_str = f.read()

        try:
            # print("Now is: ", af_pdb_filepath, "and ", gt_pdb_filepath)
            po = from_pdb_string(pdb_str=pdb_str, gt_pdb_str=gt_pdb_str, chain_id=chain, ca_only=False, full_seq_AMR=full_seq_AMR, data_type=data_type, dataset_type=dataset_type)
        except ValueError as e: 
            print("Errors: ", af_pdb_filepath, "and ", gt_pdb_filepath)
        
        if po == False:
            return False

        # extend element informations
        
        gt_atom_mask = torch.from_numpy(eval(f"po.{'gt_atom_mask'}").astype(np.float32))
        
        atom2cgids = torch.from_numpy(eval(f"po.{'atom2cgids'}").astype(np.float32))
        gt_atom_positions = torch.from_numpy(eval(f"po.{'gt_atom_positions'}").astype(np.float32))
        
        all_atom_positions = torch.from_numpy(eval(f"po.{'atom_positions'}").astype(np.float32))
        
        rmsd = torch.from_numpy(eval(f"po.{'rmsd'}").astype(np.float32))

        trans_target = torch.from_numpy(eval(f"po.{'trans_target'}").astype(np.float32))
        rot_target = torch.from_numpy(eval(f"po.{'rot_target'}").astype(np.float32))
        rot_m4_target = torch.from_numpy(eval(f"po.{'rot_m4_target'}").astype(np.float32))

        pdb_name = af_pdb_filepath

        gt_res_mask = torch.from_numpy(eval(f"po.{'gt_res_mask'}"))
        if all_atom_positions.shape[0] != gt_res_mask.shape[0]:
            return False

        po2struc = {
            'atom_positions': 'xyz',
            'atom_names': 'name',
            'elements': 'element',
            'residue_names': 'resname',
        }
        structure = {}
        
        #find the backbone atom, where the index is 0,1,2,4
        res_list = torch.zeros(37)
        res_list[0], res_list[1], res_list[2], res_list[4] = 1,1,1,1

        align_atom_positions = torch.from_numpy(eval(f"po.{'align_pos'}").astype(np.float32))
        start_atom_mask = po.atom_mask
        
        atom_mask = start_atom_mask[gt_res_mask==1].reshape(-1).astype(bool)
        atom_mask = torch.from_numpy(atom_mask)
        start_atom_mask = torch.from_numpy(start_atom_mask)

        # 获得初始化的backbone list
        start_res_num = start_atom_mask.shape[0]
        backbone_list = res_list.repeat(start_res_num, 1).view(-1)
        
        # 获取从atom mask中继续获得backbone原子的位置，其中1代表backbone原子
        finally_atom_mask = torch.logical_and(start_atom_mask.view(-1), backbone_list).int()
        align_atom_trans = torch.from_numpy(eval(f"po.{'atom_trans_align2gt'}").astype(np.float32))
        # align_atom_trans = align_atom_trans.view(-1,3)[finally_atom_mask==1]

        # judge the dataset_type
        if dataset_type == 'NB':
            cdr1_mask = torch.from_numpy(eval(f"po.{'cdr1_mask'}"))
            cdr2_mask = torch.from_numpy(eval(f"po.{'cdr2_mask'}"))
            cdr3_mask = torch.from_numpy(eval(f"po.{'cdr3_mask'}"))

            # frist select the mask from res_mask
            # cdr1_mask_by_res_mask = cdr1_mask[gt_res_mask==1]
            cdr1_mask_by_res_mask = cdr1_mask[gt_res_mask==1]
            cdr2_mask_by_res_mask = cdr2_mask[gt_res_mask==1]
            cdr3_mask_by_res_mask = cdr3_mask[gt_res_mask==1]

            # select the atom mask and backbone_list mask
            backbone_list_for_cdr = res_list.repeat(gt_atom_positions.shape[0], 1).view(-1)
            # 获取从atom mask中继续获得backbone原子的位置，其中1代表backbone原子
            finally_atom_mask_for_cdr = torch.logical_and(gt_atom_mask.view(-1), backbone_list_for_cdr).int()
            
            finally_atom_mask_reshape_r_37 = finally_atom_mask_for_cdr.reshape((-1,37))
            # set the pos =0 of backbone_reshape_r_37 if the atom not in cdr 
            # 只要计算cdr前面的和自身区域1数量即可
            cdr1_mask_backbone_atom = torch.tensor(get_cdr_backbone_mask(cdr1_mask_by_res_mask, finally_atom_mask_reshape_r_37))
            cdr2_mask_backbone_atom = torch.tensor(get_cdr_backbone_mask(cdr2_mask_by_res_mask, finally_atom_mask_reshape_r_37))
            cdr3_mask_backbone_atom = torch.tensor(get_cdr_backbone_mask(cdr3_mask_by_res_mask, finally_atom_mask_reshape_r_37))

            # cdr1_begin_index = cdr1_mask_by_res_mask.index(1) # 是以res为单位
            # cdr1_end_index = len(cdr1_mask_by_res_mask) - 1 - cdr1_mask_by_res_mask[::-1].index(1)
            
            # cdr1_before_backbone_atome_num  =  finally_atom_mask_reshape_r_37[:cdr1_begin_index, :].sum().item() # cdr1区域之前的backbone原子数量
            # cdr1_backbone_atome_num = finally_atom_mask_reshape_r_37[cdr1_begin_index:cdr1_end_index+1, :].sum().item() #cdr1区域的backbone原子数量
            
            # backbone_atom_num_list = [0] * finally_atom_mask_reshape_r_37.sum().item() #backbone all atom
            # cdr1_mask_backbone_atom[cdr1_before_backbone_atome_num: cdr1_before_backbone_atome_num + cdr1_backbone_atome_num+1] = [1] * cdr1_backbone_atome_num

        else:
            # select the atom mask and backbone_list mask
            backbone_list_for_cdr = res_list.repeat(gt_atom_positions.shape[0], 1).view(-1)
            # 获取从atom mask中继续获得backbone原子的位置，其中1代表backbone原子
            finally_atom_mask_for_cdr = torch.logical_and(gt_atom_mask.view(-1), backbone_list_for_cdr).int()
            
            finally_atom_mask_for_cdr_backbone_num =  finally_atom_mask_for_cdr.sum().item()
            cdr1_mask_backbone_atom = [0] * finally_atom_mask_for_cdr_backbone_num #backbone all atom
            cdr1_mask_backbone_atom = torch.tensor(cdr1_mask_backbone_atom)
            cdr2_mask_backbone_atom = torch.tensor(cdr1_mask_backbone_atom)
            cdr3_mask_backbone_atom = torch.tensor(cdr1_mask_backbone_atom)

        for k in [
            'atom_positions',
            'atom_names',
            'residue_names',
            'elements',
        ]:
            item = eval(f"po.{k}")
            
            if k == 'atom_positions':
                item = item.reshape(-1, item.shape[-1])
            else:
                item = item.reshape(-1, )
            item = item[finally_atom_mask==1] # mask the 0
            
            structure[po2struc[k]] = item

        atom_names = structure['name'].tolist()
        atom_numbers = torch.Tensor([atom2id[atname] for atname in atom_names])     # atom number of atoms in residues. max number is 37.

        elements = structure['element'].tolist()
        atomic_numbers = [element2atomic_numbers[element]for element in elements]
        atomic_numbers = torch.Tensor(atomic_numbers)

        res_names = structure['resname'].tolist()
        res1 = [restype_3to1[resname] for resname in res_names]
        resid = [res_type12id[res1[i]] for i in range(len(res1))]
        resid = torch.Tensor(resid)

        X = torch.from_numpy(structure['xyz'].astype(np.float32))# 获得start model的backbone

        gt_backbone_atom_positions = gt_atom_positions.view(-1,3)

        start2gt_model_tran = torch.from_numpy(eval(f"po.{'start2gt_model_tran'}").astype(np.float32))
        start2gt_model_rot = torch.from_numpy(eval(f"po.{'start2gt_model_rot'}").astype(np.float32))

        n_nodes = X.size(0)
        data = Data(
            pos=X,
            atom_numbers=atom_numbers, 
            atomic_numbers=atomic_numbers, 
            start_atom_mask=start_atom_mask,
            resid=resid, 
            n_nodes=n_nodes,
            start2gt_model_tran = start2gt_model_tran,
            start2gt_model_rot = start2gt_model_rot,
            gt_atom_positions=gt_atom_positions,
            gt_backbone_atom_positions= gt_backbone_atom_positions,
            gt_atom_mask=gt_atom_mask,
            gt_res_mask=gt_res_mask,
            cdr1_mask_backbone_atom = cdr1_mask_backbone_atom,
            cdr2_mask_backbone_atom = cdr2_mask_backbone_atom,
            cdr3_mask_backbone_atom = cdr3_mask_backbone_atom,
            all_atom_positions=all_atom_positions, # start model pos
            align_atom_positions = align_atom_positions,
            align_atom_trans = align_atom_trans,
            atom2cgids= atom2cgids,
            pdb_name = pdb_name,
            trans_target= trans_target,
            rot_target= rot_target,
            rot_m4_target = rot_m4_target,
            start_model_backbone_mask = finally_atom_mask,
            rmsd = rmsd)
    elif data_type =='inference':
        try:
            # print("Now is: ", af_pdb_filepath, "and ", gt_pdb_filepath)
            po = from_pdb_string(pdb_str=pdb_str, gt_pdb_str=None, chain_id=None, ca_only=False, full_seq_AMR=None, data_type=data_type, dataset_type=dataset_type)
        except ValueError as e: 
            print("Errors: ", af_pdb_filepath, "and ", gt_pdb_filepath)
        
        if po == False:
            return False

        
        all_atom_positions = torch.from_numpy(eval(f"po.{'atom_positions'}").astype(np.float32))
        
        pdb_name = af_pdb_filepath

        po2struc = {
            'atom_positions': 'xyz',
            'atom_names': 'name',
            'elements': 'element',
            'residue_names': 'resname',
        }
        structure = {}
        
        #find the backbone atom, where the index is 0,1,2,4
        res_list = torch.zeros(37)
        res_list[0], res_list[1], res_list[2], res_list[4] = 1,1,1,1
        
        start_atom_mask = torch.from_numpy(po.atom_mask)

        # 获得初始化的backbone list
        start_res_num = start_atom_mask.shape[0]
        backbone_list = res_list.repeat(start_res_num, 1).view(-1)
        
        # 获取从atom mask中继续获得backbone原子的位置，其中1代表backbone原子
        finally_atom_mask = torch.logical_and(start_atom_mask.view(-1), backbone_list).int()
        # align_atom_trans = align_atom_trans.view(-1,3)[finally_atom_mask==1]

        for k in [
            'atom_positions',
            'atom_names',
            'residue_names',
            'elements',
        ]:
            item = eval(f"po.{k}")
            
            if k == 'atom_positions':
                item = item.reshape(-1, item.shape[-1])
            else:
                item = item.reshape(-1, )
            item = item[finally_atom_mask==1] # mask the 0
            
            structure[po2struc[k]] = item

        atom_names = structure['name'].tolist()
        atom_numbers = torch.Tensor([atom2id[atname] for atname in atom_names])     # atom number of atoms in residues. max number is 37.

        elements = structure['element'].tolist()
        atomic_numbers = [element2atomic_numbers[element]for element in elements]
        atomic_numbers = torch.Tensor(atomic_numbers)

        res_names = structure['resname'].tolist()
        res1 = [restype_3to1[resname] for resname in res_names]
        resid = [res_type12id[res1[i]] for i in range(len(res1))]
        resid = torch.Tensor(resid)

        X = torch.from_numpy(structure['xyz'].astype(np.float32))# 获得start model的backbone



        n_nodes = X.size(0)
        data = Data(
            pos=X,
            atom_numbers=atom_numbers, 
            atomic_numbers=atomic_numbers, 
            start_atom_mask=start_atom_mask,
            resid=resid, 
            n_nodes=n_nodes,
            all_atom_positions=all_atom_positions, # start model pos
            pdb_name = pdb_name,
            start_model_backbone_mask = finally_atom_mask
            )
    
    return data


def cached_prepare_features(af_pdb, gt_pdb, chain, full_seq, data_type, dataset_type):
    return prepare_features(
        af_pdb_filepath=af_pdb, 
        gt_pdb_filepath=gt_pdb, 
        chain=chain, 
        full_seq_AMR=full_seq, 
        data_type=data_type,
        dataset_type=dataset_type
    )

from functools import lru_cache


class PDBDataset(Dataset):
    def __init__(self, csv_file, data_type='train', dataset_type=None, sample_num=-1, valid_head=-1):
        self.df = load_data(csv_file)
        if valid_head == -1: # only get train data
            if sample_num != -1 : # get sample_num train data 
                self.df = self.df.sample(n=sample_num, random_state=42)
            # otherwise get all train dataset 
        
        elif valid_head != -1: # only get valid data
            if sample_num != -1: # get valid from sub-train-dataset 
                self.df = self.df.sample(n=sample_num, random_state=42).head(valid_head)
            else: # get valid data from all train data
                self.df = self.df.sample(n=valid_head, random_state=42)

        # if sample_num != -1 and  valid_head == -1: # get train data
            # self.df = self.df.head(sample_num)
            
        # elif sample_num != -1 and  valid_head != -1: # get valid data
        #     self.df = self.df.sample(n=sample_num, random_state=42).head(valid_head)
        # else sample_num == -1 and valid_head!=-1: # get valid data
        #     self.df = self.df.sample(n=valid_head, random_state=42)
        
        
        self.data_type = data_type
        self.dataset_type = dataset_type # if dataset_type = NB, need to computer the CDR mask

        self.pdb_files = list(self.df["pdb_fpath"])
        if self.data_type!='inference':
            self.gt_pdb_files = list(self.df["pdb_fpath_gt"])
            self.chain_ids = list(self.df["chain"])
            self.full_seq_amr = list(self.df["full_seq_AMR"])

    def __len__(self):
        return len(self.pdb_files)

    @lru_cache(maxsize=128)  # 限制最多缓存128个条目
    def get_features(self, af_pdb, gt_pdb, chain, full_seq, data_type, dataset_type):
        return cached_prepare_features(af_pdb, gt_pdb, chain, full_seq, data_type, dataset_type)

    def __getitem__(self, idx):
        if self.data_type == 'inference':
            af_pdb = self.pdb_files[idx]
            if os.path.exists(af_pdb):
                data = self.get_features(af_pdb, None, None, None, self.data_type, self.dataset_type)
               
                if data:
                    return data
                else:
                    print("pdb have error!!!!!!!!!!!!!!")
                    print(af_pdb)
                    return None
            print("there have fpath no exist !!!!!!!!!!!!!!")
            print(af_pdb)
            return None
        else:
            af_pdb = self.pdb_files[idx]
            gt_pdb = self.gt_pdb_files[idx]
            chain = self.chain_ids[idx] if not pd.isna(self.chain_ids[idx]) else None
            full_seq = self.full_seq_amr[idx] if not pd.isna(self.full_seq_amr[idx]) else None
            
            if os.path.exists(af_pdb) and os.path.exists(gt_pdb):
                data = self.get_features(af_pdb, gt_pdb, chain, full_seq, self.data_type, self.dataset_type)
                # data = prepare_features(
                #     af_pdb_filepath=af_pdb, 
                #     gt_pdb_filepath=gt_pdb, 
                #     chain=chain, 
                #     full_seq_AMR=full_seq, 
                #     data_type=self.data_type
                #     )
                if data:
                    return data
                else:
                    print("pdb have error!!!!!!!!!!!!!!")
                    print(af_pdb)
                    print(gt_pdb)
                    print(full_seq)
                    return None
            print("there have fpath no exist !!!!!!!!!!!!!!")
            print(af_pdb)
            print(gt_pdb)
            return None

   
def load_data(csv_file):
    """Load the CSV data into a DataFrame."""
    return pd.read_csv(csv_file)


import torch.nn.functional as F
from scipy.linalg import logm  

# 示例旋转矩阵（真实值和预测值），这两个是3x3的正交矩阵
# R_true = torch.tensor([[0.866, -0.500, 0.0], 
#                        [0.500, 0.866, 0.0], 
#                        [0.0, 0.0, 1.0]], requires_grad=True, dtype=torch.float32)

# R_pred = torch.tensor([[0.707, -0.707, 0.0], 
#                        [0.707, 0.707, 0.0], 
#                        [0.0, 0.0, 1.0]], requires_grad=True, dtype=torch.float32)


# Quaternion
def quaternion_loss(q_true, q_pred):
    # 计算四元数的内积差异
    dot_products = torch.abs(torch.sum(q_true * q_pred, dim=-1))
    # Loss = 1 - 四元数内积
    loss = 1.0 - dot_products
    return loss.sum()  # 损失 sum() or mean()

# t_true = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# t_pred = torch.tensor([1.5, 2.5, 3.5], requires_grad=True)
import torch.nn as nn
# 使用 L2 Loss 计算平移向量之间的差异
def l2_loss(t_true, t_pred):
    return torch.norm(t_true - t_pred, p=2)


#  使用 Huber Loss 计算平移向量差异
def huber_loss(t_true, t_pred, delta=1.0):
    loss = nn.HuberLoss(delta=delta)
    return loss(t_true, t_pred)

# 计算向量夹角误差
def batch_direction_error(A, B):
    """
    Args:
        A and B:
            [n, 3],  tensor
    Return:
        angle_rad:
            [n], tensor
    """
    # 计算点积：批量方式 (element-wise sum of product along dim=1)
    dot_product = torch.sum(A * B, dim=1)
    
    # 计算范数：批量方式 (L2 norm along dim=1)
    norm_A = torch.norm(A, dim=1) + 1e-8
    norm_B = torch.norm(B, dim=1) + 1e-8
    
    # 计算余弦值，并裁剪在 [-1, 1] 范围内，避免浮点误差
    cos_theta = torch.clamp(dot_product / (norm_A * norm_B), -1.0, 1.0)
    
    # # 计算弧度制的夹角
    # angle_rad = torch.acos(cos_theta)
    
    return cos_theta

def magnitude_loss(v1, v2):
    """
    计算批次向量 v1 和 v2 的模之间的平方差损失。

    参数:
    - v1: [n, 3] 的张量, 第一个批次的向量
    - v2: [n, 3] 的张量, 第二个批次的向量
    
    返回:
    - 平均模平方差损失 (标量)
    """
    # 逐个计算每个向量的模
    mag_v1 = torch.norm(v1, p=2, dim=1)  # shape: [n]
    mag_v2 = torch.norm(v2, p=2, dim=1)  # shape: [n]

    # 计算模之间的平方差
    mod = (mag_v1 - mag_v2) ** 2

    return mod
    

def loss_function_T_backbone(align_tran, predicted_tran, cdr1_mask=None, cdr2_mask=None, cdr3_mask=None, config_runtime=None):
    """
    iter add the r and t to each atom temp. update through add T to all atom one step. 
    Args:
        align_tran:
            [batch * backbone atom, 3], tensor
        predicted_tran:
            [batch * backbone atom, 3],  tensor
        atom_mask:
            [gt_residue, 37], translation tensor
    Return:
        mse
    """
    mod = magnitude_loss(v1=predicted_tran, v2=align_tran)
    angle = 1-batch_direction_error(predicted_tran, align_tran)

    use_CDR_loss =config_runtime['use_CDR_loss']
    if use_CDR_loss == True:
        cdr1_loss_weight=config_runtime['cdr1_loss_weight']
        cdr2_loss_weight=config_runtime['cdr2_loss_weight']
        cdr3_loss_weight=config_runtime['cdr3_loss_weight']
        #采用cdr loss
        cdr_mask = cdr1_mask*cdr1_loss_weight + cdr2_mask*cdr2_loss_weight + cdr3_mask*cdr3_loss_weight
        cdr_mask[cdr_mask == 0] = 1 # 除了cdr区域，其他backbone的权重为1
        mod = (mod * cdr_mask)
        angle_loss = (angle * cdr_mask).sum()
    else: 
        angle_loss = angle.sum()
    
    if config_runtime['tran_loss_mean_or_sum'] == 'mean':
        tran_loss = torch.mean(mod)
    else:
        tran_loss = torch.sum(mod)

    return tran_loss, angle_loss



def computer_rmsd4test_only_trans(align_tran, predicted_tran, start_model_backbone_mask, init_model, start_atom_mask, gt_model, gt_res_mask, gt_atom_mask, start2gt_model_tran, start2gt_model_rot, cdr1_mask=None, cdr2_mask=None, cdr3_mask=None, config_runtime=None):
    """
    iter add the r and t to each atom temp. update through add R and T to all atom one step. 
    Args:
        
        align_tran:
            [gt res, 37, 3], tensor
        predicted_tran:
            [start backbone, 3],  tensor
        all_atom_positions:
            [start res, 37, 3],  tensor
        start_model_backbone_mask:
            [start model res * 37], tensor
        init_model:
            [start_residue, 37, 3], tensor
        start_atom_mask:
            [start residue, 37], tensor
        gt_model:
            [gt_residue, 37, 3], tensor
        gt_res_mask:
            [full residue], tensor
        atom_mask:
            [gt_residue, 37],  tensor
        gt_atom_mask:
            [gt_residue, 37], tensor
        start2gt_model_rot:
            [3,3] ,tensor
        start2gt_model_tran:
            [3], tensor

    Return:
        rmsd   
    """
    
    # 计算gt_backbone_atom_positions，并且筛选predicted_tran, init_model_backbone与其对应
    restore_predicted_tran2start_model_shape = torch.zeros(init_model.shape).to(predicted_tran.device).view(-1,3)
    #根据backbone list去按照位置填入
    count_A = start_model_backbone_mask.sum().item()  # 计算 start_model_backbone_mask 中 1 的数量
    count_B = predicted_tran.size(0)  # 获取 predicted_tran 的行数

    if count_A != count_B:
        raise ValueError(f"Error: Number of 1s in start_model_backbone_mask ({count_A}) does not match the number of rows in predicted_tran ({count_B}).")

    # 获取start_model_backbone_mask中为 1 的位置
    indices = start_model_backbone_mask.nonzero(as_tuple=True)[0]  # shape [n], n 是 A 中 1 的数量

    # 将 predicted_tran 的值赋给 restore_predicted_tran2start_model_shape 的相应位置
    restore_predicted_tran2start_model_shape[indices] = predicted_tran

    # 再根据gt res mask和 gt atom mask获取跟gt backbone相同的atom
    #find the backbone atom, where the index is 0,1,2,4
    res_list = torch.zeros(37).to(predicted_tran.device)
    res_list[0], res_list[1], res_list[2], res_list[4] = 1,1,1,1


    # 获得初始化的backbone list
    gt_res_num = gt_model.shape[0]
    backbone_list = res_list.repeat(gt_res_num, 1).view(-1)
    # 获取从gt atom mask中继续获得gt backbone原子的位置，其中1代表backbone原子
    finally_backbone_atom_mask = torch.logical_and(gt_atom_mask.view(-1), backbone_list).int()
    
    #find the CA atom, where the index is 1
    res_list_CA = torch.zeros(37).to(predicted_tran.device)
    res_list_CA[1] = 1

    # 获得初始化的backbone list
    gt_res_num = gt_model.shape[0]
    CA_list = res_list_CA.repeat(gt_res_num, 1).view(-1)
    
    # 获取从gt atom mask中继续获得gt backbone原子的位置，其中1代表backbone原子
    finally_CA_atom_mask = torch.logical_and(gt_atom_mask.view(-1), CA_list).int()
    
    # 计算start model的backbone rmsd
    rot_start, tran_start_, rmsd_start, superimpose_start = superimpose_single(
        gt_model.view(-1,3), init_model[gt_res_mask==1].view(-1,3), mask= finally_backbone_atom_mask # mask中为1则是该原子存在，需要rmsd计算
    )

    # 获得start Model 的CA rmsd
    rot_start_CA, tran_start_CA, rmsd_start_CA, superimpose_start_CA = superimpose_single(
            gt_model.view(-1,3), init_model[gt_res_mask==1].view(-1,3), mask=finally_CA_atom_mask # mask中为1则是该原子存在，需要rmsd计算
        )
    
    # 计算 refine model的backone rmsd
    refine_backbone_atom = init_model+restore_predicted_tran2start_model_shape.reshape(init_model.shape)
    
    rot_refine, tran_refine, rmsd_refine, superimpose_refine = superimpose_single(
            gt_model.view(-1,3), refine_backbone_atom[gt_res_mask==1].view(-1,3), mask=finally_backbone_atom_mask # mask中为1则是该原子存在，需要rmsd计算
        )

    # 获得refine model的 CA rmsd
    rot_refine_CA, tran_refine_CA, rmsd_refine_CA, superimpose_refine_CA = superimpose_single(
            gt_model.view(-1,3), refine_backbone_atom[gt_res_mask==1].view(-1,3), mask=finally_CA_atom_mask # mask中为1则是该原子存在，需要rmsd计算
        )

    # computer the CA loss of CDR 
    CA_in_backbone = finally_CA_atom_mask[finally_backbone_atom_mask==1] #获得backbone长度的CA mask
    cdr_mask_without_weight = cdr1_mask + cdr2_mask + cdr3_mask # backbone长度cdr mask
    # 两者取并集，获得cdr区域的CA
    cdr_CA_backbone_mask = torch.logical_and(CA_in_backbone, cdr_mask_without_weight).int()

    rot_refine_CA_CDR, tran_refine_CA_CDR, rmsd_refine_CA_CDR_after, superimpose_refine_CA_CDR = superimpose_single(
            gt_model.view(-1,3)[finally_backbone_atom_mask==1], refine_backbone_atom[gt_res_mask==1].view(-1,3)[finally_backbone_atom_mask==1], mask= cdr_CA_backbone_mask# mask中为1则是该原子存在，需要rmsd计算
        )

    rot_refine_CA_CDR_before, tran_refine_CA_CDR_before, rmsd_refine_CA_CDR_before, superimpose_refine_CA_CDR_before = superimpose_single(
            gt_model.view(-1,3)[finally_backbone_atom_mask==1], init_model[gt_res_mask==1].view(-1,3)[finally_backbone_atom_mask==1], mask=cdr_CA_backbone_mask # mask中为1则是该原子存在，需要rmsd计算
        )

    # 将预测的tran旋转到align的位置上
    rot_predicted_tran =(torch.matmul(start2gt_model_rot, restore_predicted_tran2start_model_shape.T).T)
    # rot_predicted_tran =(torch.matmul(start2gt_model_rot, predicted_tran.T).T)

    rot_predicted_tran_mask = rot_predicted_tran.reshape(init_model.shape)[gt_res_mask==1].reshape((-1,3))[finally_backbone_atom_mask==1]
    
    align_backbone_tran = align_tran.reshape((-1,3))[finally_backbone_atom_mask==1]

    mod = magnitude_loss(rot_predicted_tran_mask, align_backbone_tran)
    angle = 1-batch_direction_error(rot_predicted_tran_mask, align_backbone_tran)

    # mod = magnitude_loss(v1=predicted_tran, v2=align_tran)
    # angle = 1-batch_direction_error(predicted_tran, align_tran)
    use_CDR_loss =config_runtime['use_CDR_loss']
    if use_CDR_loss == True:
        #采用cdr loss
        cdr1_loss_weight=config_runtime['cdr1_loss_weight']
        cdr2_loss_weight=config_runtime['cdr2_loss_weight']
        cdr3_loss_weight=config_runtime['cdr3_loss_weight']

        cdr_mask = cdr1_mask*cdr1_loss_weight + cdr2_mask*cdr2_loss_weight + cdr3_mask*cdr3_loss_weight
        cdr_mask[cdr_mask == 0] = 1 # 除了cdr区域，其他backbone的权重为1
        mod = (mod * cdr_mask)
        angle_loss = (angle * cdr_mask).sum()
        


    else: 
        angle_loss = angle.sum()

    if config_runtime['tran_loss_mean_or_sum'] == 'mean':
        tran_loss = torch.mean(mod)
    else:
        tran_loss = torch.sum(mod)
    # tran_loss = loss_mse(rot_predicted_tran.reshape((-1,3))[finally_backbone_atom_mask==1], align_tran.reshape((-1,3))[finally_backbone_atom_mask==1])

    return rmsd_refine, rmsd_start, rmsd_refine_CA, rmsd_start_CA, tran_loss, angle_loss, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after




# def loss_function_R_T(gt_m4_rot, gt_rot, gt_tran, gt_res_mask, rot_m4, tran):
#     """
#     iter add the r and t to each atom temp. update through add R and T to all atom one step. 
#     Args:
#         gt_m4_rot:
#             [gt_residue, 37, 4], tensor
#         gt_tran:
#             [gt_residue, 37, 3], tensor
#         rot_m4:
#             [af_residue, 37, 4], rotation tensor
#         tran:
#             [af_residue, 37, 3], translation tensor
#         gt_res_mask:
#             [gt_residue], tensor
      
        
#     Return:
#         rmsd
            
#     """
#     # 采用einsum
#     # rot = torch.rand(2,5,3,3)
#     # coords = torch.rand(2,5,3)
#     # tran = torch.rand(2,5, 3)
#     refine_rot_m4 = rot_m4[gt_res_mask==1]
#     refine_tran = tran[gt_res_mask==1]
    
#     # rot_quaternion_loss = quaternion_loss(gt_m4_rot, refine_rot_m4)
#     rot_m4_mse_loss = loss_mse(gt_m4_rot, refine_rot_m4)

#     tran_mse_loss = loss_mse(gt_tran, refine_tran)
    
#     return rot_m4_mse_loss, tran_mse_loss



# def computer_rmsd4test(gt_m4_rot, gt_rot, gt_tran, init_model, gt_model, gt_res_mask, gt_atom_mask, rot_m4, tran, atom2cgids):
#     """
#     iter add the r and t to each atom temp. update through add R and T to all atom one step. 
#     Args:
#         init_model:
#             [batch*residue, 37, 3], tensor
#         gt_model:
#             [batch*residue, 37, 3], tensor
#         rot:
#             [batch*residue, 37, 3, 3], rotation tensor
#         tran:
#             [batch*residue, 37, 3], translation tensor
#         gt_res_mask:
#             [batch*residue], tensor
#         gt_atom_mask:
#             [batch*gt_residue, 37], tensor
        
#     Return:
#         rmsd
            
#     """
#     # 采用einsum
#     # rot = torch.rand(2,5,3,3)
#     # coords = torch.rand(2,5,3)
#     # tran = torch.rand(2,5, 3)

#     # computer the mse 
#     refine_rot_m4 = rot_m4[gt_res_mask==1]
#     refine_tran = tran[gt_res_mask==1]
    
#     rot_m4_mse_loss = loss_mse(gt_m4_rot, refine_rot_m4)
#     tran_mse_loss = loss_mse(gt_tran, refine_tran)
    


#     each_res_atom_times = torch.sum(atom2cgids, dim=-1)
#     # 转为形状 [batch_size, residue，atom=37, 1] 以进行广播
#     t_last = each_res_atom_times.unsqueeze(-1)
#     # 防止除以 0，将 t_last 中的 0 替换为一个很小的值（例如1e-6），避免计算错误
#     t_last_safe = torch.where(t_last == 0, torch.tensor(1e-6).to(tran.device), t_last)
#     # x 的最后一个维度除以 t 的最后一个值
#     mean_cg_trans_atom = atom2cgids / t_last_safe

#     # first get the R [3,3] from m4 [4]
#     # 找到所有行都为 0 的行
#     rot_m4 = rot_m4.view(-1,4)
#     zero_rows = (rot_m4 == 0).all(dim=1)

#     # 将这些行的第一个元素设置为 1
#     rot_m4[zero_rows, 0] = 1.0
#     cg_rot = torch.from_numpy(R.from_quat(rot_m4.cpu().numpy()).as_matrix()).to(tran.device).to(torch.float32)

#     # 然后乘以输出，获得每个原子的平均值,R and T
#     # cg_rot = cg_rot.view(-1,4,3,3)
#     atom_rot = torch.matmul(mean_cg_trans_atom, cg_rot.view(-1,4,9)).view(mean_cg_trans_atom.shape[0],mean_cg_trans_atom.shape[1],3,3)[gt_res_mask==1]
#     atom_tran = torch.matmul(mean_cg_trans_atom, tran)[gt_res_mask==1]

#     init_model_res_mask = init_model[gt_res_mask==1] # get the start model by res mask

#     # add pred_R_T to start model, then computer the rmsd
#     refine_model = torch.einsum("brij, brj -> bri", atom_rot, init_model_res_mask) + atom_tran

#     gt_model_reshape = gt_model.view(-1,3)

#     refine_model_reshape = refine_model.view(-1, 3)
#     trans_gt_mask = gt_atom_mask.view(-1)
#     rot_, tran_, rmsd_add_Pred_RT2init, superimpose = superimpose_single(
#             gt_model_reshape, refine_model_reshape, mask=trans_gt_mask # mask中为1则是该原子存在，需要rmsd计算
#         )

#     # add pred_R_T to gt model, then computer the rmsd
#     # refine_add_pred_R_T2gt_model = torch.einsum("brij, brj -> bri", atom_rot, gt_model) + atom_tran
#     # rot_, tran_, rmsd_add_Pred_RT2gt, superimpose = superimpose_single(
#     #         init_model_res_mask.view(-1,3), refine_add_pred_R_T2gt_model.view(-1,3), mask=trans_gt_mask # mask中为1则是存在，不用mask
#     #     )


#     # add the gt_m4_R and gt_T to start model, then computer rmsd between them, atom level
#     gt_m4_rot = gt_m4_rot.view(-1,4)
#     zero_rows = (gt_m4_rot == 0).all(dim=1)
#     gt_m4_rot[zero_rows, 0] = 1.0

#     gt_cg_rot = torch.from_numpy(R.from_quat(gt_m4_rot.cpu().numpy()).as_matrix()).to(tran.device).to(torch.float32)
#     gt_mean_cg_trans_atom = mean_cg_trans_atom[gt_res_mask==1]


#     gt_atom_m4_rot = torch.matmul(gt_mean_cg_trans_atom, gt_cg_rot.view(-1,4,9)).view(gt_mean_cg_trans_atom.shape[0],gt_mean_cg_trans_atom.shape[1],3,3)
#     gt_atom_tran = torch.matmul(gt_mean_cg_trans_atom, gt_tran)


#     refine_init_model_bygt = torch.einsum("brij, brj -> bri", gt_atom_m4_rot, init_model_res_mask) + gt_atom_tran
#     refine_init_model_after_atom_mask_bygt = refine_init_model_bygt.view(-1,refine_init_model_bygt.shape[-1])

#     rot_, tran_, rmsd_add_gt_m4_RT2init, superimpose = superimpose_single(
#                 gt_model_reshape, refine_init_model_after_atom_mask_bygt, mask=trans_gt_mask # mask中为1则是存在，不用mask
#             )


#     # add the gt_R and gt_T to start model, then computer rmsd between them
#     gt_atom_rot = torch.matmul(gt_mean_cg_trans_atom, gt_rot.view(-1,4,9)).view(gt_mean_cg_trans_atom.shape[0],gt_mean_cg_trans_atom.shape[1],3,3)

#     refine_init_model_bygt = torch.einsum("brij, brj -> bri", gt_atom_rot, init_model_res_mask) + gt_atom_tran
#     refine_init_model_after_atom_mask_bygt = refine_init_model_bygt.view(-1,refine_init_model_bygt.shape[-1])
#     rot_, tran_, rmsd_add_gt_RT2init, superimpose = superimpose_single(
#                 gt_model_reshape, refine_init_model_after_atom_mask_bygt, mask=trans_gt_mask # mask中为1则是存在，不用mask
#             )

#     # add the gt_R and gt_T to gt model, then computer rmsd between them
#     refine_init_model_bygt = torch.einsum("brij, brj -> bri", gt_atom_rot, gt_model) + gt_atom_tran
#     refine_init_model_after_atom_mask_bygt = refine_init_model_bygt.view(-1,refine_init_model_bygt.shape[-1])

#     rot_, tran_, rmsd_add_gt_RT2gt, superimpose = superimpose_single(
#                 init_model_res_mask.view(-1,3), refine_init_model_after_atom_mask_bygt, mask=trans_gt_mask # mask中为1则是存在，不用mask
#             )

#     # add the gt_m4_R and gt_T to gt model, then computer rmsd between them

#     refine_init_model_bygt = torch.einsum("brij, brj -> bri", gt_atom_m4_rot, gt_model) + gt_atom_tran
#     refine_init_model_after_atom_mask_bygt = refine_init_model_bygt.view(-1,refine_init_model_bygt.shape[-1])
    
#     rot_, tran_, rmsd_add_gt_m4_RT2gt, superimpose = superimpose_single(
#                 init_model_res_mask.view(-1,3), refine_init_model_after_atom_mask_bygt, mask=trans_gt_mask # mask中为1则是存在，不用mask
#             )
#     return rmsd_add_Pred_RT2init, rmsd_add_gt_RT2init, rmsd_add_gt_m4_RT2init, rot_m4_mse_loss, tran_mse_loss

# 自定义学习率规则，根据训练进程动态调整
def lr_lambda(current_step):
    # 示例：在每 200 个 batch 后
    if current_step % 200 == 0:
        return 0.99  # 乘以 0.99
    return 1.0  # 不变

# 使用 LambdaLR

def data_list_collater(data_list):
    """Run data_list_collater method."""
    # code.
    data_list = [data for data in data_list if data is not None]
    print("data_list_collater!!!!!!!!!!")
    print(data_list)
    if len(data_list) == 0:
        return None
    if len(data_list[0]) == 2:
        data_list1 = [data[0] for data in data_list]
        data_list2 = [data[1] for data in data_list]
        batch1 = Batch.from_data_list(data_list1)
        batch2 = Batch.from_data_list(data_list2)
        return [batch1, batch2]
    elif len(data_list[0]) == 1:
        data_list1 = [data[0] for data in data_list]
        batch1 = Batch.from_data_list(data_list1)
        return [batch1]
    else:
        raise RuntimeError(f"Unsupported data list collater! data_list: {data_list}")


def test_run_train(args, config_runtime, config_model, config_data):
    
    wandb_PJ_name = args.wandb_name
    experiment_name = args.experiment_name
    # local_rank = args.local_rank
    local_rank = args.rank

    # local_rank = args.rank
    prefetch_factor = config_data['prefetch_factor'] 
    batch_size = config_data['trn_batch_size'] 
    epoch = config_runtime['epoch']
    lr = config_runtime['learning_rate']
    sample_num = config_data['sample_num'] 
    valid_head = config_data['valid_head'] 
    min_num_workers = config_data['min_num_workers'] 
    log_interval = config_runtime['log_interval']
    log_interval_for_test = config_runtime['log_interval_for_test']
    weight_tran =  config_runtime['weight_tran']
    weight_angle = config_runtime['weight_angle']
    weight_angle_decay_rate = config_runtime['weight_angle_decay_rate']

    use_CDR_loss =config_runtime['use_CDR_loss']
    cdr1_loss_weight=config_runtime['cdr1_loss_weight']
    cdr2_loss_weight=config_runtime['cdr2_loss_weight']
    cdr3_loss_weight=config_runtime['cdr3_loss_weight']

    dataset_type = config_data['dataset_type']

    set_random_seed(config_runtime['seed'])
    # get the pdb fpath
    train_file_path = config_data['train_filepath'][0]['path']
    train_file_name = train_file_path.split("/")[-1].split(".")[0]
  
    test_file_path = config_data['test_filepath']
    
    

    num_workers = min(min_num_workers, os.cpu_count() // dist_utils.get_world_size())
    
    
    train_dataset = PDBDataset(csv_file=train_file_path, data_type='train', dataset_type=dataset_type, sample_num=sample_num, valid_head=-1)
    valid_dataset = PDBDataset(csv_file=train_file_path, data_type='test', dataset_type=dataset_type, sample_num=sample_num, valid_head=valid_head)
    test_dataset = PDBDataset(csv_file=test_file_path, data_type='test', dataset_type=dataset_type, sample_num=-1, valid_head=-1)

    
    print('total_train_data:', len(train_dataset))
    print('total_valid_data:' , len(valid_dataset))
    print('total_test_data:' , len(test_dataset))

    train_sampler = None
    test_sampler = None
    valid_sampler = None
    if dist_utils.get_world_size() > 1:
        print('We will training on gpu:', local_rank)
        train_sampler = DistributedSampler(
            train_dataset,
            rank=local_rank,
            num_replicas=dist_utils.get_world_size(),
            shuffle=True,
        )

        valid_sampler = DistributedSampler(
            valid_dataset,
            rank=local_rank,
            num_replicas=dist_utils.get_world_size(),
            shuffle=False,
        )

        test_sampler = DistributedSampler(
            test_dataset,
            rank=local_rank,
            num_replicas=dist_utils.get_world_size(),
            shuffle=False,
        )

    def custom_collate_fn(data_list):
        # 过滤掉 None 值
        data_list = [data for data in data_list if data is not None]
        # 将过滤后的数据组合成一个批次（可以根据具体需求进行处理）
        if not data_list:
            return None

        return data_list  #将数据进一步处理成批量形式


    valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle= False,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            sampler=valid_sampler,
        )

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=num_workers,
            shuffle= False,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            sampler=test_sampler,
        )
    
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True if train_sampler is None else False,
            collate_fn=custom_collate_fn,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=prefetch_factor,
            sampler=train_sampler,
            persistent_workers=True,
        )
    
    print('train_dataloader length:', len(train_dataloader))
    print('valid_dataloader length:', len(valid_dataloader))
    print('test_dataloader length:', len(test_dataloader))
    
    model_time_flag = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    
    # device = torch.device(f'cuda:{rank}')
    model = EquiformerV2(**config_model).to(local_rank)
    if config_model['model_reload_path'] != False:
        model.load_state_dict(torch.load(config_model['model_reload_path']))
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    epochs = epoch
    
    # if local_rank == 0:
    #     wandb_config = wandb.init(
    #         project=wandb_PJ_name, 
    #         name=experiment_name+'_'+str(model_time_flag +'refine_' + str(local_rank)),
    #         dir="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/logs",
    #         config={
    #         "architecture": "equiformer2",
    #         "epochs": epoch,
    #         "train_data_file": train_file_path,
    #         "test_data_file": test_file_path,
    #         }
    #     )

    loss_file = "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/rmsd_values_"+experiment_name+'_'+str(model_time_flag+'_refine_')+str(local_rank)+".txt"
    with open(loss_file, "w") as file:
        file.write("Step, start pdb name, rmsd_start_CA, rmsd_refine_CA, rmsd_CA_improve_rat, start rmsd, refine rmsd, improve rat, start CDR CA rmsd, refine CDR CA rmsd \n")  
    
    # print('To train')
    step = 0
    improvement_pdb_num_all_epoch = 0
    for epoch in range(epochs):
        model.train()
        train_angle_loss = 0
        train_tran_loss = 0
        # train_dataloader.sampler.set_epoch(epoch)
        # print('the train_dataloader batch num is ', len(train_dataloader))
        accumulation_steps = 4  # update weight each accumulation_steps
        optimizer.zero_grad()
        # for data in train_dataloader:
        for i, data in enumerate(train_dataloader):
            if data is None:
                continue
            
            # we just need data.n_nodes，data.pos，data.atomic_numbers，data.atom_numbers，data.resid，data.batch
            # align_tran=data.align_atom_trans.to(local_rank)
            # init_model_backbone=data.pos.to(local_rank)
            # init_model=data.all_atom_positions.to(local_rank)
            # gt_backbone_atom_positions= data.gt_backbone_atom_positions.to(local_rank)
            # gt_model=data.gt_atom_positions.to(local_rank)
            
            predicted_atom_trans  = model(data.to(local_rank)) # align model
            
            tran_loss, angle_loss = loss_function_T_backbone(
                align_tran=data.align_atom_trans.to(local_rank), 
                predicted_tran=predicted_atom_trans,
                cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
                config_runtime=config_runtime
                )

            loss = weight_tran * tran_loss + weight_angle * angle_loss
            
            loss = loss / accumulation_steps  # grad norm
            # optimizer.zero_grad()
            loss.backward()
            # optimizer.step()

            # update weight each accumulation_steps 
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                optimizer.step()  # update
                # update the lr
                scheduler.step()
                optimizer.zero_grad()  # zero grad

            train_tran_loss += weight_tran * tran_loss.detach() 
            train_angle_loss += weight_angle * angle_loss.detach() 

            # log the training loss each log_interval 
            # log the training loss each log_interval_for_test 
            if step % log_interval_for_test == 0  and step % log_interval == 0:
                # update the loss weight, 如果两loss之间相差一个数量级
                if torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) > 1 and torch.abs(train_angle_loss)>torch.abs(train_tran_loss):
                    weight_angle = weight_angle * 0.9
                elif torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) > 1 and torch.abs(train_angle_loss) < torch.abs(train_tran_loss):
                    weight_tran = weight_tran * 0.9
            
                # log the train info!
                avg_tran_loss = train_tran_loss.item() / (log_interval * batch_size)
                avg_angle_loss = train_angle_loss.item() / (log_interval * batch_size)
                train_angle_loss = 0 # reset 
                train_tran_loss = 0

                # log the test info!
                improve_rat_all = 0.0
                all_pdb_improve_rat = 0.0
                improvement_pdb_num = 0
                CDR_loss_all_pdb_rat = 0.0
                test_loss = 0.0
                tran_mean_loss = 0.0
                angle_mean_loss = 0.0
                valid_tran_mean_loss = 0.0
                valid_angle_mean_loss = 0.0
                model.eval()
                # print('To test')
                test_last_pdb_name= None
                test_last_pdb_trans_output = None

                with torch.no_grad():
                    for data in tqdm(test_dataloader):

                        predicted_atom_trans  = model(data.to(local_rank))
                        pdb_name = data.pdb_name[0]
                        if pdb_name == '/nfs_beijing/liyakun/folding_model_store/models/nb_models/3loss_ssfape0.1_wCdr_nb_distill_v2.3_fix/predict/8op0_A/ranked_unrelax.pdb':
                            test_last_pdb_name= pdb_name
                            test_last_pdb_trans_output = predicted_atom_trans

                        rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, rot_pred_tran_mse, rot_pred_angle, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after = computer_rmsd4test_only_trans(
                            align_tran=data.align_atom_trans.to(local_rank),
                            predicted_tran=predicted_atom_trans, 
                            start_model_backbone_mask= data.start_model_backbone_mask.to(local_rank),
                            init_model=data.all_atom_positions.to(local_rank), 
                            start_atom_mask=data.start_atom_mask.to(local_rank), 
                            gt_model=data.gt_atom_positions.to(local_rank), 
                            gt_res_mask=data.gt_res_mask.to(local_rank), 
                            gt_atom_mask=data.gt_atom_mask.to(local_rank),
                            start2gt_model_tran = data.start2gt_model_tran.to(local_rank),
                            start2gt_model_rot = data.start2gt_model_rot.to(local_rank),
                            cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                            cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                            cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
                            config_runtime=config_runtime
                            )
                        
                        
                        CA_improve_rat = (rmsd_start_CA.item() - rmsd_refine_CA.item())/ rmsd_start_CA.item()
                        all_pdb_improve_rat += CA_improve_rat

                        CDR_CA_improve_rat = (rmsd_refine_CA_CDR_before.item() - rmsd_refine_CA_CDR_after.item())/ rmsd_refine_CA_CDR_before.item()

                        backbone_improve_rat = (rmsd_start.item() - rmsd.item())/ rmsd_start.item()

                        with open(loss_file, "a") as file:
                            file.write(f"Step:{step}, {pdb_name}, {rmsd_start_CA}, {rmsd_refine_CA}, {CA_improve_rat}, {rmsd_start}, {rmsd}, {backbone_improve_rat}, {rmsd_refine_CA_CDR_before}, {rmsd_refine_CA_CDR_after}\n")
                        
                        

                        if rmsd_start_CA.item() - rmsd_refine_CA.item() > 0: # if after refine rmsd is better than start
                            # print('start RMSD:', rmsd_start.item())
                            # print('refine RMSD:', rmsd.item())
                            # print('pred_tran_mse_loss:', rot_pred_tran_mse.item())
                            # print('pred_angle_loss:', rot_pred_angle.item())
                            # print('---------')
                            improvement_pdb_num += 1
                            improve_rat_all += CA_improve_rat
                        
                        tran_mean_loss += weight_tran * rot_pred_tran_mse.detach().item()
                        angle_mean_loss += weight_angle * rot_pred_angle.detach().item()

                    for data in valid_dataloader:
    
                        predicted_atom_trans  = model(data.to(local_rank))
                      
                        rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, rot_pred_tran_mse, rot_pred_angle, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after = computer_rmsd4test_only_trans(
                            align_tran=data.align_atom_trans.to(local_rank),
                            predicted_tran=predicted_atom_trans, 
                            start_model_backbone_mask= data.start_model_backbone_mask.to(local_rank),
                            init_model=data.all_atom_positions.to(local_rank), 
                            start_atom_mask=data.start_atom_mask.to(local_rank), 
                            gt_model=data.gt_atom_positions.to(local_rank), 
                            gt_res_mask=data.gt_res_mask.to(local_rank), 
                            gt_atom_mask=data.gt_atom_mask.to(local_rank),
                            start2gt_model_tran = data.start2gt_model_tran.to(local_rank),
                            start2gt_model_rot = data.start2gt_model_rot.to(local_rank),
                            cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                            cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                            cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
                            config_runtime=config_runtime
                            )
                        
                      
                        valid_tran_mean_loss += weight_tran*rot_pred_tran_mse.detach().item()
                        valid_angle_mean_loss += weight_angle*rot_pred_angle.detach().item()

                    # log metrics to wandb
                    metrics = {
                        "CA_improvement_mean_rat": improve_rat_all / (improvement_pdb_num+1e-8),
                        "CA_improvement_pdb_num": improvement_pdb_num,
                        "all_pdb_improve_rat": all_pdb_improve_rat/len(test_dataloader),
                        "CDR_CA_improve_rat": CDR_CA_improve_rat/len(test_dataloader),
                        "test_tran_mean_loss": tran_mean_loss / len(test_dataloader),
                        "test_angle_mean_loss": angle_mean_loss / len(test_dataloader),
                        "valid_tran_mean_loss": valid_tran_mean_loss / len(valid_dataloader),
                        "valid_angle_mean_loss": valid_angle_mean_loss / len(valid_dataloader),
                        "avg_tran_loss_log_interval": avg_tran_loss,
                        "avg_angle_loss_log_interval": avg_angle_loss
                    }
                    metrics_total = {name: [] for name in metrics}
                    for name, value in metrics.items():
                        torch.distributed.barrier()  # 同步所有进程

                        metrics_gathered = [None for _ in range(dist_utils.get_world_size())]  # 存储各进程数据
                        torch.distributed.all_gather_object(metrics_gathered, value)  # 收集所有 GPU 的指标

                        if local_rank == 0:  # 仅在 rank 0 上进行汇总
                            metrics_total[name].extend(metrics_gathered)  # 将收集到的数据添加到 metrics_total

                    # 在 rank 0 上一次性上传所有指标到 W&B
                    if local_rank == 0:
                        averaged_metrics = {}  # 用于存储最终处理后的指标
                        for name, values in metrics_total.items():
                            if name == "CA_improvement_pdb_num":
                                # 对 CA_improvement_pdb_num 进行累加
                                averaged_metrics[name] = sum(values)
                            else:
                                # 对其他指标取平均值
                                averaged_metrics[name] = sum(values) / len(values)

                        # 将所有平均值打包成一个字典，并上传到 W&B
                        averaged_metrics["step"] = step  # 将 step 加入字典
                        # wandb_config.log(averaged_metrics)  # 一次性上传所有指标
                    
                        if averaged_metrics['CA_improvement_mean_rat'] > improvement_pdb_num_all_epoch:
                            improvement_pdb_num_all_epoch = averaged_metrics['CA_improvement_mean_rat']
                            torch.save(model.module.state_dict(), "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/"+experiment_name+'_'+str(model_time_flag)+'_Step_'+str(step)+'_improve_rat_'+str(averaged_metrics['CA_improvement_mean_rat'])+'_imporve_num_'+str(averaged_metrics['CA_improvement_pdb_num'])+".ckpt")
                            #exit
                            if step > 0:
                                # save the last pdb info
                                # 将字符串和 NumPy 数组保存到txt 文件
                                with open('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/rmsd_values_'+experiment_name+'_'+str(model_time_flag+'_refine_'+str(step))+'_sin_check_trans_output_last_pdb.txt', 'w') as f:
                                    # 先写入字符串
                                    f.write(test_last_pdb_name + '\n')
                                    # 写入张量数据
                                    np.savetxt(f, test_last_pdb_trans_output.cpu().numpy())
                                sys.exit()


            elif step % log_interval == 0 :
                # update the loss weight
                if torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) > 1 and torch.abs(train_angle_loss) > torch.abs(train_tran_loss):
                    weight_angle = weight_angle * 0.9
                elif torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) > 1 and torch.abs(train_angle_loss) < torch.abs(train_tran_loss):
                    weight_tran = weight_tran * 0.9
            
                # print('log the train info!')
                avg_tran_loss = train_tran_loss.item() / (log_interval * batch_size)
                avg_angle_loss = train_angle_loss.item() / (log_interval * batch_size)
                
                train_angle_loss = 0 # reset 
                train_tran_loss = 0

                metrics = {
                        "avg_tran_loss_log_interval": avg_tran_loss,
                        "avg_angle_loss_log_interval": avg_angle_loss
                    }

                metrics_total = {name: [] for name in metrics}
                for name, value in metrics.items():
                    torch.distributed.barrier()  # 同步所有进程

                    metrics_gathered = [None for _ in range(dist_utils.get_world_size())]  # 存储各进程数据
                    torch.distributed.all_gather_object(metrics_gathered, value)  # 收集所有 GPU 的指标

                    if local_rank == 0:  # 仅在 rank 0 上进行汇总
                        metrics_total[name].extend(metrics_gathered)  # 将收集到的数据添加到 metrics_total

                # 在 rank 0 上一次性上传所有指标到 W&B
                if local_rank == 0:
                    # 计算每个指标的平均值
                    averaged_metrics = {name: sum(values) / len(values) for name, values in metrics_total.items()}

                    # 将所有平均值打包成一个字典，并上传到 W&B
                    averaged_metrics["step"] = step  # 将 step 加入字典
                    # wandb_config.log(averaged_metrics)  # 一次性上传所有指标
                      

            step += 1



# def main_worker(rank, nprocs, args, train_file_path, test_file_path):
#     ## DDP：DDP init
    
#     local_rank = rank
#     torch.cuda.set_device(local_rank)
#     dist.init_process_group(backend='nccl',init_method=args.dist_url,
#                         world_size=args.nprocs,
#                         rank=rank)
    
    
#     dataset_type = 'NB'
#     cdr1_loss_weight = args.cdr1_loss_weight
#     cdr2_loss_weight = args.cdr2_loss_weight
#     cdr3_loss_weight = args.cdr3_loss_weight

#     args.num_workers = min(args.min_num_workers, os.cpu_count() // dist_utils.get_world_size())
#     print('we use num_workers:', args.num_workers)
#     print('dist_utils.get_world_size()', dist_utils.get_world_size())
#     print('dist_utils.get_rank()', dist_utils.get_rank())
#     print('rank:',rank)
    
    
#     train_dataset = PDBDataset(csv_file=train_file_path, data_type='train', dataset_type=dataset_type, sample_num=args.sample_num, valid_head=-1)
#     valid_dataset = PDBDataset(csv_file=train_file_path, data_type='test', dataset_type=dataset_type, sample_num=args.sample_num, valid_head=args.valid_head)
#     test_dataset = PDBDataset(csv_file=test_file_path, data_type='test', dataset_type=dataset_type, sample_num=-1, valid_head=-1)

    
#     print('total_train_data:', len(train_dataset))
#     print('total_valid_data:' , len(valid_dataset))
#     print('total_test_data:' , len(test_dataset))

#     train_sampler = None
#     test_sampler = None
#     if dist_utils.get_world_size() > 1:
#         print('We will training on gpu:', local_rank)
#         train_sampler = DistributedSampler(
#             train_dataset,
#             rank=local_rank,
#             num_replicas=dist_utils.get_world_size(),
#             shuffle=True,
#         )

#         valid_sampler = DistributedSampler(
#             valid_dataset,
#             rank=local_rank,
#             num_replicas=dist_utils.get_world_size(),
#             shuffle=False,
#         )

#         test_sampler = DistributedSampler(
#             test_dataset,
#             rank=local_rank,
#             num_replicas=dist_utils.get_world_size(),
#             shuffle=False,
#         )

#     valid_dataloader = DataLoader(
#             valid_dataset,
#             batch_size=1,
#             num_workers=args.num_workers,
#             shuffle= False,
#             collate_fn=data_list_collater,
#             pin_memory=True,
#             # prefetch_factor=args.prefetch_factor,
#             sampler=valid_sampler,
#         )

#     test_dataloader = DataLoader(
#             test_dataset,
#             batch_size=1,
#             num_workers=args.num_workers,
#             shuffle= False,
#             collate_fn=data_list_collater,
#             pin_memory=True,
#             # prefetch_factor=args.prefetch_factor,
#             sampler=test_sampler,
#         )
    
#     train_dataloader = DataLoader(
#             train_dataset,
#             batch_size=args.batch_size,
#             # batch_size=32,
#             num_workers=args.num_workers,
#             shuffle=True if train_sampler is None else False,
#             collate_fn=data_list_collater,
#             drop_last=True,
#             pin_memory=True,
#             prefetch_factor=args.prefetch_factor,
#             sampler=train_sampler,
#             persistent_workers=True,
#         )
    
    

#     print('train_dataloader length:', len(train_dataloader))
#     print('valid_dataloader length:', len(valid_dataloader))
#     print('test_dataloader length:', len(test_dataloader))
    
#     model_time_flag = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    
#     # device = torch.device(f'cuda:{rank}')
#     model = EquiformerV2().to(local_rank)
    
#     model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

#     optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     # 自定义学习率规则，根据训练进程动态调整
#     def lr_lambda(current_step):
#         # 示例：在每 200 个 batch 后
#         if current_step % 200 == 0:
#             return 0.99  # 乘以 0.99
#         return 1.0  # 不变

#     # 使用 LambdaLR
#     scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
#     epochs = args.epoch

#     # for i, data in enumerate(train_dataloader):
#     #         if data is None:
#     #             continue
            
#     #         predicted_atom_trans  = model(data.to(local_rank)) # align model
#     #         tran_loss, angle_loss = loss_function_T_backbone(
#     #             align_tran=data.align_atom_trans.to(local_rank), 
#     #             predicted_tran=predicted_atom_trans,
#     #             cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
#     #             cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
#     #             cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
#     #             cdr1_loss_weight=cdr1_loss_weight, 
#     #             cdr2_loss_weight=cdr2_loss_weight, 
#     #             cdr3_loss_weight=cdr3_loss_weight
#     #             )
    
#     # test test_dataloader
#     # for i, data in enumerate(test_dataloader): 
#     #     if data is None:
#     #         continue
#     #     predicted_atom_trans  = model(data.to(local_rank))
#     #     pdb_name = data.pdb_name
#     #     print(pdb_name[0])
#     #     rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, rot_pred_tran_mse, rot_pred_angle = computer_rmsd4test_only_trans(
#     #         align_tran=data.align_atom_trans.to(local_rank),
#     #         predicted_tran=predicted_atom_trans, 
#     #         start_model_backbone_mask= data.start_model_backbone_mask.to(local_rank),
#     #         init_model=data.all_atom_positions.to(local_rank), 
#     #         start_atom_mask=data.start_atom_mask.to(local_rank), 
#     #         gt_model=data.gt_atom_positions.to(local_rank), 
#     #         gt_res_mask=data.gt_res_mask.to(local_rank), 
#     #         gt_atom_mask=data.gt_atom_mask.to(local_rank),
#     #         start2gt_model_tran = data.start2gt_model_tran.to(local_rank),
#     #         start2gt_model_rot = data.start2gt_model_rot.to(local_rank))
    
#     if local_rank == 0:
#         wandb_config = wandb.init(
#             project="test-project", 
#             name=str(model_time_flag +'refine_' + str(local_rank)),
#             config={
#             "architecture": "equiformer2",
#             "epochs": args.epoch,
#             "train_data_file": train_file_path,
#             "test_data_file": test_file_path,
#             }
#         )

#     loss_file = "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/rmsd_values_"+str(model_time_flag+'_refine_')+str(local_rank)+".txt"
#     with open(loss_file, "w") as file:
#         file.write("Step, start pdb name, rmsd_start_CA, rmsd_refine_CA, rmsd_CA_improve_rat, start rmsd, refine rmsd, improve rat,  rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after \n")  
    
#     # print('To train')
#     step = 0
#     improvement_pdb_num_all_epoch = 0
#     for epoch in range(epochs):
#         model.train()
#         train_angle_loss = 0
#         train_tran_loss = 0
#         # train_dataloader.sampler.set_epoch(epoch)
#         # print('the train_dataloader batch num is ', len(train_dataloader))
#         accumulation_steps = 4  # update weight each accumulation_steps
#         optimizer.zero_grad()
#         # for data in train_dataloader:
#         for i, data in enumerate(train_dataloader):
#             if data is None:
#                 continue
            
#             # we just need data.n_nodes，data.pos，data.atomic_numbers，data.atom_numbers，data.resid，data.batch
#             # align_tran=data.align_atom_trans.to(local_rank)
#             # init_model_backbone=data.pos.to(local_rank)
#             # init_model=data.all_atom_positions.to(local_rank)
#             # gt_backbone_atom_positions= data.gt_backbone_atom_positions.to(local_rank)
#             # gt_model=data.gt_atom_positions.to(local_rank)
            
#             predicted_atom_trans  = model(data.to(local_rank)) # align model
#             tran_loss, angle_loss = loss_function_T_backbone(
#                 align_tran=data.align_atom_trans.to(local_rank), 
#                 predicted_tran=predicted_atom_trans,
#                 cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
#                 cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
#                 cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
#                 cdr1_loss_weight=cdr1_loss_weight, 
#                 cdr2_loss_weight=cdr2_loss_weight, 
#                 cdr3_loss_weight=cdr3_loss_weight
#                 )
      
#             loss = args.weight_tran * tran_loss + args.weight_angle * angle_loss
            
#             loss = loss / accumulation_steps  # grad norm
#             # optimizer.zero_grad()
#             loss.backward()
#             # optimizer.step()

#             # update weight each accumulation_steps 
#             if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
#                 optimizer.step()  # update
#                 # update the lr
#                 scheduler.step()
#                 optimizer.zero_grad()  # zero grad
            
            

#             train_tran_loss += args.weight_tran * tran_loss.detach() 
#             train_angle_loss += args.weight_angle * angle_loss.detach() 

#             # log the training loss each log_interval 
#             # log the training loss each log_interval_for_test 
#             if step % args.log_interval_for_test == 0  and step % args.log_interval == 0:
#                 # update the loss weight
#                 if torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) < 1 and torch.abs(train_angle_loss)>torch.abs(train_tran_loss):
#                     args.weight_angle = args.weight_angle * 0.9
#                 elif torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) < 1 and torch.abs(train_angle_loss) < torch.abs(train_tran_loss):
#                     args.weight_tran = args.weight_tran * 0.9
            
#                 # log the train info!
#                 avg_tran_loss = train_tran_loss.item() / (args.log_interval * args.batch_size)
#                 avg_angle_loss = train_angle_loss.item() / (args.log_interval * args.batch_size)
#                 train_angle_loss = 0 # reset 
#                 train_tran_loss = 0

#                 # log the test info!'
#                 improve_rat_all = 0.0
#                 improvement_pdb_num = 0
#                 test_loss = 0.0
#                 tran_mean_loss = 0.0
#                 angle_mean_loss = 0.0
#                 valid_tran_mean_loss = 0.0
#                 valid_angle_mean_loss = 0.0
#                 model.eval()
#                 # print('To test')
#                 with torch.no_grad():
#                     for data in tqdm(test_dataloader):

#                         predicted_atom_trans  = model(data.to(local_rank))
#                         pdb_name = data.pdb_name[0]
#                         rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, rot_pred_tran_mse, rot_pred_angle, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after = computer_rmsd4test_only_trans(
#                             align_tran=data.align_atom_trans.to(local_rank),
#                             predicted_tran=predicted_atom_trans, 
#                             start_model_backbone_mask= data.start_model_backbone_mask.to(local_rank),
#                             init_model=data.all_atom_positions.to(local_rank), 
#                             start_atom_mask=data.start_atom_mask.to(local_rank), 
#                             gt_model=data.gt_atom_positions.to(local_rank), 
#                             gt_res_mask=data.gt_res_mask.to(local_rank), 
#                             gt_atom_mask=data.gt_atom_mask.to(local_rank),
#                             start2gt_model_tran = data.start2gt_model_tran.to(local_rank),
#                             start2gt_model_rot = data.start2gt_model_rot.to(local_rank),
#                             cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
#                             cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
#                             cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
#                             cdr1_loss_weight=cdr1_loss_weight, 
#                             cdr2_loss_weight=cdr2_loss_weight, 
#                             cdr3_loss_weight=cdr3_loss_weight
#                             )
                        
                        
#                         CA_improve_rat = (rmsd_start_CA.item() - rmsd_refine_CA.item())/ rmsd_start_CA.item()
#                         backbone_improve_rat = (rmsd_start.item() - rmsd.item())/ rmsd_start.item()

#                         with open(loss_file, "a") as file:
#                             file.write(f"Step:{step}, {pdb_name}, {rmsd_start_CA:.4f}, {rmsd_refine_CA:.4f}, {CA_improve_rat:.4f}, {rmsd_start:.4f}, {rmsd:.4f}, {backbone_improve_rat:.4f}\n")
                        
#                         if rmsd_start_CA.item() - rmsd_refine_CA.item() > 0: # if after refine rmsd is better than start
#                             # print('start RMSD:', rmsd_start.item())
#                             # print('refine RMSD:', rmsd.item())
#                             # print('pred_tran_mse_loss:', rot_pred_tran_mse.item())
#                             # print('pred_angle_loss:', rot_pred_angle.item())
#                             # print('---------')
#                             improvement_pdb_num += 1
#                             improve_rat_all += CA_improve_rat
                        
#                         tran_mean_loss += rot_pred_tran_mse.detach().item()
#                         angle_mean_loss += rot_pred_angle.detach().item()

#                     for data in valid_dataloader:
    
#                         predicted_atom_trans  = model(data.to(local_rank))
                      
#                         rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, rot_pred_tran_mse, rot_pred_angle , rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after= computer_rmsd4test_only_trans(
#                             align_tran=data.align_atom_trans.to(local_rank),
#                             predicted_tran=predicted_atom_trans, 
#                             start_model_backbone_mask= data.start_model_backbone_mask.to(local_rank),
#                             init_model=data.all_atom_positions.to(local_rank), 
#                             start_atom_mask=data.start_atom_mask.to(local_rank), 
#                             gt_model=data.gt_atom_positions.to(local_rank), 
#                             gt_res_mask=data.gt_res_mask.to(local_rank), 
#                             gt_atom_mask=data.gt_atom_mask.to(local_rank),
#                             start2gt_model_tran = data.start2gt_model_tran.to(local_rank),
#                             start2gt_model_rot = data.start2gt_model_rot.to(local_rank),
#                             cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
#                             cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
#                             cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
#                             cdr1_loss_weight=cdr1_loss_weight, 
#                             cdr2_loss_weight=cdr2_loss_weight, 
#                             cdr3_loss_weight=cdr3_loss_weight
#                             )
                        
                      
#                         valid_tran_mean_loss += rot_pred_tran_mse.detach().item()
#                         valid_angle_mean_loss += rot_pred_angle.detach().item()

#                     # log metrics to wandb
#                     metrics = {
#                         "CA_improvement_mean_rat": improve_rat_all / improvement_pdb_num,
#                         "CA_improvement_pdb_num": improvement_pdb_num,
#                         "test_tran_mean_loss": tran_mean_loss / len(test_dataloader),
#                         "test_angle_mean_loss": angle_mean_loss / len(test_dataloader),
#                         "valid_tran_mean_loss": valid_tran_mean_loss / len(valid_dataloader),
#                         "valid_angle_mean_loss": valid_angle_mean_loss / len(valid_dataloader),
#                         "avg_tran_loss_log_interval": avg_tran_loss,
#                         "avg_angle_loss_log_interval": avg_angle_loss
#                     }
#                     metrics_total = {name: [] for name in metrics}
#                     for name, value in metrics.items():
#                         torch.distributed.barrier()  # 同步所有进程

#                         metrics_gathered = [None for _ in range(dist_utils.get_world_size())]  # 存储各进程数据
#                         torch.distributed.all_gather_object(metrics_gathered, value)  # 收集所有 GPU 的指标

#                         if local_rank == 0:  # 仅在 rank 0 上进行汇总
#                             metrics_total[name].extend(metrics_gathered)  # 将收集到的数据添加到 metrics_total

#                     # 在 rank 0 上一次性上传所有指标到 W&B
#                     if local_rank == 0:
#                         averaged_metrics = {}  # 用于存储最终处理后的指标
#                         for name, values in metrics_total.items():
#                             if name == "CA_improvement_pdb_num":
#                                 # 对 CA_improvement_pdb_num 进行累加
#                                 averaged_metrics[name] = sum(values)
#                             else:
#                                 # 对其他指标取平均值
#                                 averaged_metrics[name] = sum(values) / len(values)

#                         # 将所有平均值打包成一个字典，并上传到 W&B
#                         averaged_metrics["step"] = step  # 将 step 加入字典
#                         wandb_config.log(averaged_metrics)  # 一次性上传所有指标
                    
#                         if averaged_metrics['CA_improvement_mean_rat'] > improvement_pdb_num_all_epoch:
#                             improvement_pdb_num_all_epoch = averaged_metrics['CA_improvement_mean_rat']
#                             torch.save(model.module.state_dict(), "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/"+str(model_time_flag)+'_Step_'+str(step)+'_improve_rat_'+str(averaged_metrics['CA_improvement_mean_rat'])+'_imporve_num_'+str(averaged_metrics['CA_improvement_pdb_num'])+".ckpt")
            
#             elif step % args.log_interval == 0 :
#                 # update the loss weight
#                 if torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) < 1 and torch.abs(train_angle_loss)>torch.abs(train_tran_loss):
#                     args.weight_angle = args.weight_angle * 0.9
#                 elif torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) < 1 and torch.abs(train_angle_loss) < torch.abs(train_tran_loss):
#                     args.weight_tran = args.weight_tran * 0.9
            
#                 # print('log the train info!')
#                 avg_tran_loss = train_tran_loss.item() / (args.log_interval * args.batch_size)
#                 avg_angle_loss = train_angle_loss.item() / (args.log_interval * args.batch_size)
                
#                 train_angle_loss = 0 # reset 
#                 train_tran_loss = 0

#                 metrics = {
#                         "avg_tran_loss_log_interval": avg_tran_loss,
#                         "avg_angle_loss_log_interval": avg_angle_loss
#                     }

#                 metrics_total = {name: [] for name in metrics}
#                 for name, value in metrics.items():
#                     torch.distributed.barrier()  # 同步所有进程

#                     metrics_gathered = [None for _ in range(dist_utils.get_world_size())]  # 存储各进程数据
#                     torch.distributed.all_gather_object(metrics_gathered, value)  # 收集所有 GPU 的指标

#                     if local_rank == 0:  # 仅在 rank 0 上进行汇总
#                         metrics_total[name].extend(metrics_gathered)  # 将收集到的数据添加到 metrics_total

#                 # 在 rank 0 上一次性上传所有指标到 W&B
#                 if local_rank == 0:
#                     # 计算每个指标的平均值
#                     averaged_metrics = {name: sum(values) / len(values) for name, values in metrics_total.items()}

#                     # 将所有平均值打包成一个字典，并上传到 W&B
#                     averaged_metrics["step"] = step  # 将 step 加入字典
#                     wandb_config.log(averaged_metrics)  # 一次性上传所有指标
                      

#             step += 1
        
      


"""
the model default setting:
    use_pbc=False,
    regress_forces=False,
    otf_graph=True,
    max_neighbors=20,
    max_radius=5.0,
    max_num_elements=90,
    max_num_atom_names=37,
    max_num_residues=21,

    num_layers=4,
    sphere_channels=16,
    attn_hidden_channels=8,
    num_heads=4,
    attn_alpha_channels=16,
    attn_value_channels=8,
    ffn_hidden_channels=8,

    norm_type='layer_norm_sh',

    lmax_list=[2],
    mmax_list=[2],
    grid_resolution=18,

    num_sphere_samples=16,

    edge_channels=8,
    use_atom_edge_embedding=True,
    share_atom_edge_embedding=False,
    use_m_share_rad=False,
    distance_function="gaussian",
    num_distance_basis=512,

    attn_activation='silu',
    use_s2_act_attn=False,
    use_attn_renorm=True,
    ffn_activation='silu',
    use_gate_act=False,
    use_grid_mlp=True,
    use_sep_s2_act=True,

    alpha_drop=0.1,
    drop_path_rate=0.05,
    proj_drop=0.0,

    weight_init='uniform',
"""

# def main(args_from_input=None, config_runtime=None, config_model=None, config_data=None):
#     ## DDP：从外部得到local_rank参数。从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数
#     print('args_from_input:',args_from_input)
#     print('config_runtime:', config_runtime)
#     print('config_model:', config_model)
#     print('config_data:', config_data)
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", default=-1, type=int)
   

#     args = parser.parse_args()
#     port_id = 20000 + np.random.randint(0, 10000)
#     args.dist_url = 'tcp://127.0.0.1:' + str(port_id) #'8007' 
#     args.nprocs = torch.cuda.device_count()
  
#     if args_from_input != None:
#         args.prefetch_factor = config_data['prefetch_factor'] 
#         args.batch_size = config_data['trn_batch_size'] 
#         args.epoch = config_runtime['epoch']
#         args.lr = config_runtime['learning_rate']
#         args.sample_num = -1 # if you need all data, set sample_num = -1
#         args.valid_head = 100 
#         args.min_num_workers = 4
#         args.log_interval = config_runtime['log_interval']
#         args.log_interval_for_test = config_runtime['log_interval_for_test']
#         args.weight_tran =  config_runtime['weight_tran']
#         args.weight_angle = config_runtime['weight_angle']
#         args.weight_angle_decay_rate = config_runtime['weight_angle_decay_rate']
#     else:
#         args.prefetch_factor = 1
#         args.batch_size = 8
#         args.epoch = 100
#         args.lr = 0.001
#         args.sample_num = -1 # if you need all data, set sample_num = -1
#         args.valid_head = 100 
#         args.min_num_workers = 4
#         args.log_interval = 200
#         args.log_interval_for_test = 2000
#         args.weight_tran = 1
#         args.weight_angle = 1
#         args.weight_angle_decay_rate = 0.99
#         args.cdr1_loss_weight= 2
#         args.cdr2_loss_weight= 2
#         args.cdr3_loss_weight= 4
        
#     print(args)
#     # set random seed
#     set_random_seed(13)
#     # get the pdb fpath
#     train_file_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset_valid.csv'
#     #NB_train_dataset_valid数据为去除掉不存在pdb文件之后的可用数据。

    
#     test_file_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522.csv'
    
    
#     print('ready to main_worker!!!!!!!')
#     mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args, train_file_path, test_file_path))
    

# if __name__ == '__main__':
#     main()
    # add a inference function
    
    