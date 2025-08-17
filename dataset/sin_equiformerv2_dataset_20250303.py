import sys
sys.path.append('/nfs_beijing_ai/jinxian/rama-scoring1.3.0')
import argparse
import os
os.environ["WANDB_IGNORE_GIT"] = "1"

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
import time
from np.protein import from_pdb_string, update_pdb_with_new_coords

from dataset.feature.featurizer import process
from utils.data_encoding import encode_structure, encode_features, extract_topology
from dataset.screener import UpperboundScreener, ProbabilityScreener1d
from utils.logger import Logger
from utils.constants.atom_constants import *
from utils.constants.residue_constants import *
from utils.constants.cg_constants import *
from utils.bin_design import *
import np.residue_constants as rc
import math
logger = Logger.logger
"""Protein data type."""
import dataclasses
import io
import re
import collections
from typing import Any, Mapping, Optional, Sequence
import numpy as np
# from np import residue_constants
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

from torch.profiler import profile, record_function, ProfilerActivity

import logging
from datetime import datetime

os.environ["WANDB_MODE"] = "run"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 定义EMA模型
class EMA:
    """对模型参数进行EMA的辅助类"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = model
        # 使用clone来拷贝当前模型参数
        self.shadow = {name: param.detach().clone() 
                       for name, param in model.state_dict().items() 
                       if param.requires_grad}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.state_dict().items():
            if name in self.shadow and param.requires_grad:
                self.shadow[name].mul_(self.decay).add_((1 - self.decay) * param)

    @torch.no_grad()
    def apply_to(self, model):
        """将EMA参数复制到指定模型中"""
        for name, param in model.state_dict().items():
            if name in self.shadow and param.requires_grad:
                param.copy_(self.shadow[name])

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
    
    if data_type =='train': #输入模型的节点是来自align model
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

            cdr_mask_by_res_mask = cdr1_mask_by_res_mask +cdr2_mask_by_res_mask+cdr3_mask_by_res_mask

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

        # 保存每个氨基酸的名字
        res_names_list = []
        atom_res_names_matrix = eval(f"po.{'residue_names'}")
        for a_res in atom_res_names_matrix:
            for i in range(len(a_res)):
                if a_res[i] != None:
                    static_res_num_begin = a_res_1st_cg_num[a_res[i]]
                    res_names_list.append(static_res_num_begin)
                    break
        res_names_list = torch.tensor(res_names_list)

        gt_backbone_atom_positions = gt_atom_positions.view(-1,3)[finally_atom_mask==1]
        X = align_atom_positions.view(-1,3)[finally_atom_mask==1]
        n_nodes = X.size(0)

        
        # 计算distmap所需的cdr mask
        # gt_backbone_atom = gt_atom_positions.view(-1, 3)[finally_atom_mask == 1]
        # N = gt_backbone_atom.size(0)
        # gt_distmap = torch.cdist(gt_backbone_atom, gt_backbone_atom)
        
        # # 生成上三角索引
        # rows, cols = torch.triu_indices(N, N, offset=1, device=gt_distmap.device)
        # distmap_cdr1_mask = cdr1_mask_backbone_atom[rows] & cdr1_mask_backbone_atom[cols]
        # distmap_cdr2_mask = cdr2_mask_backbone_atom[rows] & cdr2_mask_backbone_atom[cols]
        # distmap_cdr3_mask = cdr3_mask_backbone_atom[rows] & cdr3_mask_backbone_atom[cols]
        atom_mask_align2gt = torch.from_numpy(eval(f"po.{'atom_mask_align2gt'}"))

        data = Data(
            pos=X,
            atom_numbers=atom_numbers, 
            atomic_numbers=atomic_numbers, 
            start_atom_mask=start_atom_mask,
            resid=resid, 
            n_nodes=n_nodes,
            res_names_list=res_names_list,
            gt_atom_positions=gt_atom_positions,
            gt_backbone_atom_positions = gt_backbone_atom_positions,
            gt_atom_mask=gt_atom_mask,
            gt_res_mask=gt_res_mask,
            cdr_mask_by_res_mask=cdr_mask_by_res_mask,
            cdr1_mask_backbone_atom = cdr1_mask_backbone_atom,
            cdr2_mask_backbone_atom = cdr2_mask_backbone_atom,
            cdr3_mask_backbone_atom = cdr3_mask_backbone_atom,
            # distmap_cdr1_mask = distmap_cdr1_mask,
            # distmap_cdr2_mask = distmap_cdr2_mask,
            # distmap_cdr3_mask = distmap_cdr3_mask,
            all_atom_positions=all_atom_positions, # start model pos
            atom_mask_align2gt = atom_mask_align2gt,
            align_atom_positions = align_atom_positions,
            align_atom_trans = align_atom_trans,
            atom2cgids= atom2cgids,
            pdb_name = pdb_name,
            trans_target= trans_target,
            rot_target= rot_target,
            rot_m4_target = rot_m4_target,
            start_model_backbone_mask = finally_atom_mask,
            rmsd = rmsd)

    elif data_type =='test': # 注意X应该是将start model的gt res mask backbone都输入，而不是输入align model
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
        gt_res_num = align_atom_positions.shape[0]
        backbone_list = res_list.repeat(gt_res_num, 1).view(-1)
        
        # 获取从atom mask中继续获得backbone原子的位置，其中1代表backbone原子
        finally_atom_mask = torch.logical_and(atom_mask.view(-1), backbone_list).int()
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
            cdr_mask_by_res_mask = cdr1_mask_by_res_mask +cdr2_mask_by_res_mask+cdr3_mask_by_res_mask

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
            cdr1_mask_backbone_atom = torch.tensor(cdr1_mask_backbone_atom).clone().detach() 
            cdr2_mask_backbone_atom = torch.tensor(cdr1_mask_backbone_atom).clone().detach() 
            cdr3_mask_backbone_atom = torch.tensor(cdr1_mask_backbone_atom).clone().detach() 

        for k in [
            'atom_positions',
            'atom_names',
            'residue_names',
            'elements',
        ]:
            item = eval(f"po.{k}")
            
            if k == 'atom_positions':
                item = item[gt_res_mask==1].reshape(-1, item.shape[-1])
            else:
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

        # 保存每个氨基酸的名字
        res_names_list = []
        atom_res_names_matrix = eval(f"po.{'residue_names'}")
        for a_res in atom_res_names_matrix:
            for i in range(len(a_res)):
                if a_res[i] != None:
                    static_res_num_begin = a_res_1st_cg_num[a_res[i]]
                    res_names_list.append(static_res_num_begin)
                    break
        res_names_list = torch.tensor(res_names_list)

        X = torch.from_numpy(structure['xyz'].astype(np.float32))# 获得start model的backbone

        gt_backbone_atom_positions = gt_atom_positions.view(-1,3)

        start2gt_model_tran = torch.from_numpy(eval(f"po.{'start2gt_model_tran'}").astype(np.float32))
        start2gt_model_rot = torch.from_numpy(eval(f"po.{'start2gt_model_rot'}").astype(np.float32))

        n_nodes = X.size(0)
        # pos=X,
            # atom_numbers=atom_numbers, 
            # atomic_numbers=atomic_numbers, 
            # start_atom_mask=start_atom_mask,
            # resid=resid, 
            # n_nodes=n_nodes,
            # all_atom_positions=all_atom_positions, # start model pos
            # pdb_name = pdb_name,
            # start_model_backbone_mask = finally_atom_mask
        
        # 计算distmap所需的cdr mask
        # gt_backbone_atom = gt_atom_positions.view(-1, 3)[finally_atom_mask == 1]
        # N = gt_backbone_atom.size(0)
        # gt_distmap = torch.cdist(gt_backbone_atom, gt_backbone_atom)
        
        # # 生成上三角索引
        # rows, cols = torch.triu_indices(N, N, offset=1, device=gt_distmap.device)
        # distmap_cdr1_mask = cdr1_mask_backbone_atom[rows] & cdr1_mask_backbone_atom[cols]
        # distmap_cdr2_mask = cdr2_mask_backbone_atom[rows] & cdr2_mask_backbone_atom[cols]
        # distmap_cdr3_mask = cdr3_mask_backbone_atom[rows] & cdr3_mask_backbone_atom[cols]
        atom_mask_align2gt = torch.from_numpy(eval(f"po.{'atom_mask_align2gt'}"))
        data = Data(
            pos=X,
            atom_numbers=atom_numbers, 
            atomic_numbers=atomic_numbers, 
            start_atom_mask=start_atom_mask,
            resid=resid, 
            n_nodes=n_nodes,
            res_names_list=res_names_list,
            start2gt_model_tran = start2gt_model_tran,
            start2gt_model_rot = start2gt_model_rot,
            gt_atom_positions=gt_atom_positions,
            gt_backbone_atom_positions= gt_backbone_atom_positions,
            gt_atom_mask=gt_atom_mask,
            gt_res_mask=gt_res_mask,
            cdr_mask_by_res_mask=cdr_mask_by_res_mask,
            cdr1_mask_backbone_atom = cdr1_mask_backbone_atom,
            cdr2_mask_backbone_atom = cdr2_mask_backbone_atom,
            cdr3_mask_backbone_atom = cdr3_mask_backbone_atom,
            atom_mask_align2gt = atom_mask_align2gt,
            # distmap_cdr1_mask = distmap_cdr1_mask,
            # distmap_cdr2_mask = distmap_cdr2_mask,
            # distmap_cdr3_mask = distmap_cdr3_mask,
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

        
        self.data_type = data_type
        self.dataset_type = dataset_type # if dataset_type = NB, need to computer the CDR mask

        
        if self.data_type=='train':
            # self.RMSD_CA = list(self.df["RMSD_CA"])
            # self.CDR3_RMSD_CA = list(self.df["CDR3_RMSD_CA"])
            # select
            self.df = self.df[(self.df['RMSD_CA'] <= 5) & (self.df['CDR3_RMSD_CA'] <= 10)]
            self.pdb_files = list(self.df["pdb_fpath"])
            self.gt_pdb_files = list(self.df["pdb_fpath_gt"])
            self.chain_ids = list(self.df["chain"])
            self.full_seq_amr = list(self.df["full_seq_AMR"])
        elif self.data_type=='test':
            self.pdb_files = list(self.df["pdb_fpath"])
            self.gt_pdb_files = list(self.df["pdb_fpath_gt"])
            self.chain_ids = list(self.df["chain"])
            self.full_seq_amr = list(self.df["full_seq_AMR"])
        else:
            self.pdb_files = list(self.df["pdb_fpath"])

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
            if os.path.exists(af_pdb) and os.path.exists(gt_pdb):
                chain = self.chain_ids[idx] if not pd.isna(self.chain_ids[idx]) else None
                full_seq = self.full_seq_amr[idx] if not pd.isna(self.full_seq_amr[idx]) else None

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
                    return None
                    # if idx > 1:
                    #     idx = idx - 1
                    # else:
                    #     idx = len(self.pdb_files) - 1
                        
                # print("there have fpath no exist !!!!!!!!!!!!!!")
                # print(af_pdb)
                # print(gt_pdb)
                # return None

   
def load_data(csv_file):
    """Load the CSV data into a DataFrame."""
    return pd.read_csv(csv_file)


import torch.nn.functional as F
from scipy.linalg import logm  

import torch.nn as nn





def create_nonuniform_bin_edges():
    """创建非均匀的bin边界"""
    edges = []
    # 2.0~8.0, step=0.25
    dist = 2.0
    while dist < 8.0:
        dist += 0.25
        edges.append(dist)
    
    # 8.0~20.0, step=1.0
    dist = 8.0
    while dist < 20.0:
        dist += 1.0
        edges.append(dist)
        
    return edges

# 全局常量，避免重复计算
BIN_EDGES = torch.tensor(create_nonuniform_bin_edges())
MIN_DIST = 2.0
MAX_DIST = 20.0

def get_expected_distance(logits, bin_edges):
    """计算期望距离"""
    device = logits.device
    
    # 预计算representative distances
    rep_distances = torch.cat([
        torch.tensor([MIN_DIST], device=device),
        (torch.tensor([MIN_DIST], device=device) + bin_edges[0]) / 2,
        (bin_edges[:-1] + bin_edges[1:]) / 2,
        torch.tensor([MAX_DIST + 1.0], device=device)
    ])
    
    probs = F.softmax(logits, dim=-1)
    expected_dist = (probs * rep_distances).sum(dim=-1)
    return expected_dist

def pair_idx_from_ij(i, j, N):
    """计算配对索引"""
    less_mask = (i < j)
    ii = torch.where(less_mask, i, j)
    jj = torch.where(less_mask, j, i)
    
    ii = ii.long()
    jj = jj.long()
    term1 = (2*N - ii - 1) * ii // 2
    pair_idx = term1 + (jj - ii - 1)
    return pair_idx

def get_d(i, j, expected_dist, N):
    """获取距离值"""
    p_idx = pair_idx_from_ij(i, j, N)
    return expected_dist[p_idx]


def precompute_cdr_sparse_indices(cdr1_mask, cdr2_mask, cdr3_mask, device):
    cdr_nodes = [
        torch.where(mask.to(device))[0]
        for mask in [cdr1_mask, cdr2_mask, cdr3_mask]
    ]
    
    sparse_indices_list = []
    for nodes in cdr_nodes:
        if len(nodes) < 2:
            continue
        pairs = torch.combinations(nodes, 2, with_replacement=False)
        sparse_indices_list.append(pairs.t().contiguous())
    
    if not sparse_indices_list:
        return None
    return torch.cat(sparse_indices_list, dim=1).to(device)


def triangle_inequality_loss(expected_dist, N, sample_triplets=500, node_indices=None):
    # device = expected_dist.device
    # if node_indices is None:
    #     node_indices = torch.arange(N, device=device)
    
    # num_nodes = len(node_indices)
    # if num_nodes < 3:
    #     return torch.tensor(0.0, device=device)
    
    # # 生成所有可能的三元组索引
    # # idx = torch.randint(0, num_nodes, (3 * sample_triplets,), device=device)
    # # i = node_indices[idx[:sample_triplets]]
    # # j = node_indices[idx[sample_triplets: 2*sample_triplets]]
    # # k = node_indices[idx[2*sample_triplets:]]
    # idx = torch.combinations(torch.arange(num_nodes, device=device), r=3)  # 生成所有组合
    # i, j, k = idx[:,0], idx[:,1], idx[:,2]
    
    # mask = (i != j) & (j != k) & (i != k)
    # i, j, k = i[mask], j[mask], k[mask]
    # if i.numel() == 0:
    #     return torch.tensor(0.0, device=device)
    
    # # 动态计算一维索引获取距离值
    # d_ij = expected_dist[pair_idx_from_ij(i, j, N)]
    # d_ik = expected_dist[pair_idx_from_ij(i, k, N)]
    # d_jk = expected_dist[pair_idx_from_ij(j, k, N)]
    
    # violations = F.relu(d_ij - d_ik - d_jk) + F.relu(d_ik - d_ij - d_jk) + F.relu(d_jk - d_ij - d_ik)

    nodes = node_indices if node_indices is not None else torch.arange(N)
    
    # 预计算所有可能的三元组组合（避免动态采样）
    idx = torch.combinations(nodes, r=3)  # 生成所有组合 (i,j,k)
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    
    # 向量化计算所有三元组索引
    idx_ij = pair_idx_from_ij(i, j, N)
    idx_ik = pair_idx_from_ij(i, k, N)
    idx_jk = pair_idx_from_ij(j, k, N)
    
    # 批量获取距离值
    d_ij = expected_dist[idx_ij]
    d_ik = expected_dist[idx_ik]
    d_jk = expected_dist[idx_jk]
    
    # 计算违反三角不等式的情况
    violations = F.relu(d_ij - d_ik - d_jk) + F.relu(d_ik - d_ij - d_jk) + F.relu(d_jk - d_ij - d_ik)
    return violations.mean()



def distmap_loss_function(pred_distmap, gt_model, start_model_backbone_mask, distmap_loss_fn, 
                        cdr1_mask, cdr2_mask, cdr3_mask, 
                        # distmap_cdr1_mask,
                        # distmap_cdr2_mask,
                        # distmap_cdr3_mask, 
                        use_cdr_mask=False, sample_triplets=2000, cdr_weight=[1,1,10]):
    # pred_distmap 是一个N*(N-1)/2的向量
    # gt_model是[res, 37, 3]
    # start_model_backbone_mask 是[res*37]
    # 提取骨干原子坐标
    gt_backbone_atom = gt_model.view(-1, 3)[start_model_backbone_mask == 1]
    N = gt_backbone_atom.size(0)
    gt_distmap = torch.cdist(gt_backbone_atom, gt_backbone_atom)
    
    # # 生成上三角索引
    rows, cols = torch.triu_indices(N, N, offset=1, device=gt_distmap.device)
    true_distances = gt_distmap[rows, cols]
    
    # 计算分箱索引
    bin_edges = BIN_EDGES.to(pred_distmap.device)
    bin_indices = torch.bucketize(true_distances, bin_edges)
    final_bins = torch.where(true_distances <= 2.0, 0, 
                            torch.where(true_distances > 20.0, len(bin_edges) + 1, bin_indices + 1))
    
    # 计算期望距离（保持一维数组）
    expected_dist = get_expected_distance(pred_distmap, bin_edges)
    
    class_loss = torch.tensor(0.0, device=pred_distmap.device)
    tri_loss = torch.tensor(0.0, device=pred_distmap.device)
    # cdr1_class_loss, cdr2_class_loss,cdr3_class_loss = torch.tensor(0.0, device=pred_distmap.device),torch.tensor(0.0, device=pred_distmap.device),torch.tensor(0.0, device=pred_distmap.device)
    # cdr1_tri_loss,cdr2_tri_loss,cdr3_tri_loss = torch.tensor(0.0, device=pred_distmap.device),torch.tensor(0.0, device=pred_distmap.device),torch.tensor(0.0, device=pred_distmap.device)
    if use_cdr_mask:
        try:
            # 合并所有 CDR 区域掩码
            combined_mask = cdr1_mask + cdr2_mask + cdr3_mask
            
            # 分类损失：统一处理所有 CDR 区域
            mask = combined_mask[rows] & combined_mask[cols]
            mask = mask.to(pred_distmap.device)
            
            sparse_indices = torch.nonzero(mask).squeeze()  # 转换为稀疏索引           
            class_loss = distmap_loss_fn(pred_distmap[sparse_indices], final_bins[sparse_indices])
            
            # 三角损失：统一采样所有 CDR 节点
            nodes = torch.where(combined_mask)[0]
            tri_loss = triangle_inequality_loss(expected_dist, N, sample_triplets, nodes)
        except Exception as e:
            print("Pred_distmap shape:", pred_distmap.shape)
            print("final_bins shape:", final_bins.shape)
            print("combined_mask shape:", combined_mask.shape)
            print("mask shapes:", mask.shape)
            print("rows shapes:", rows.shape)

        # if distmap_cdr1_mask.any():
        #     sparse_indices = torch.nonzero(distmap_cdr1_mask).squeeze()  # 转换为稀疏索引
        #     cdr1_class_loss = distmap_loss_fn(pred_distmap[sparse_indices], final_bins[sparse_indices])
        
        #     # 三角损失：采样所有 CDR 节点
        #     nodes = torch.where(distmap_cdr1_mask)[0]
        #     cdr1_tri_loss = triangle_inequality_loss(expected_dist, N, sample_triplets, nodes)   

        # if distmap_cdr2_mask.any():
        #     sparse_indices = torch.nonzero(distmap_cdr2_mask).squeeze()  # 转换为稀疏索引
        #     cdr2_class_loss = distmap_loss_fn(pred_distmap[sparse_indices], final_bins[sparse_indices])
        
        #     # 三角损失：采样所有 CDR 节点
        #     nodes = torch.where(distmap_cdr2_mask)[0]
        #     cdr2_tri_loss = triangle_inequality_loss(expected_dist, N, sample_triplets, nodes)   

        # if distmap_cdr3_mask.any():
        #     sparse_indices = torch.nonzero(distmap_cdr3_mask).squeeze()  # 转换为稀疏索引
        #     cdr3_class_loss = distmap_loss_fn(pred_distmap[sparse_indices], final_bins[sparse_indices])
        
        #     # 三角损失：采样所有 CDR 节点
        #     nodes = torch.where(distmap_cdr3_mask)[0]
        #     cdr3_tri_loss = triangle_inequality_loss(expected_dist, N, sample_triplets, nodes)   

        # class_loss = cdr1_class_loss*cdr_weight[0] + cdr1_class_loss*cdr_weight[1] + cdr1_class_loss*cdr_weight[2]
        # tri_loss = cdr1_tri_loss*cdr_weight[0] + cdr2_tri_loss*cdr_weight[1] + cdr3_tri_loss*cdr_weight[2]
    else:
    
        # 全局分类损失
        class_loss = distmap_loss_fn(pred_distmap, final_bins)
        # 全局三角损失（使用所有节点）
        tri_loss = triangle_inequality_loss(expected_dist, N, sample_triplets)
    
    return class_loss, tri_loss


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
    

#先获得static_cg
s_cg = load_static_cg('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/utils/resources/cg_X0.npz').clone().detach()
def cg_loss_function(refine_backbone_atom, res_names_list):
    backbone_mask = torch.zeros(refine_backbone_atom.shape[1]).to(refine_backbone_atom.device)
    backbone_mask[0],backbone_mask[1],backbone_mask[2] = 1,1,1
    static_mask = torch.zeros(s_cg.shape[1]).to(refine_backbone_atom.device)
    #循环每个res id，然后将对应的refine_backbone获取，再计算每个res的superimpose
    
    rmsd_all = 0
    for i,res_id in enumerate(res_names_list):
        # res3 = res_name
        # the_static_begin_num = a_res_1st_cg_num[res3]
        # if res3 == 'GLY':
        if res_id == 21:
            static_mask[0], static_mask[1], static_mask[2]=1,1,1
            static_cg = s_cg.to(refine_backbone_atom.device)[res_id][static_mask==1]
        else:#只取C CA和N
            static_mask[0], static_mask[1], static_mask[3]=1,1,1
            static_cg = s_cg.to(refine_backbone_atom.device)[res_id][static_mask==1]

        #获得refine res的cg
        refine_res_cg = refine_backbone_atom[i][backbone_mask==1]

        _, _, rmsd, _ = superimpose_single(
            refine_res_cg.view(-1,3), static_cg.view(-1,3), mask= torch.tensor([1,1,1]).to(refine_backbone_atom.device) # mask中为1则是该原子存在，需要rmsd计算
        )
        static_mask.fill_(0)
        rmsd_all+=rmsd
    return rmsd_all

def loss_function_T_backbone(align_tran, predicted_tran, res_names_list, gt_res_mask, gt_atom_mask, gt_model, start_model_backbone_mask, init_model, align_atom_positions, atom_mask_align2gt, cdr_mask_by_res_mask=None, cdr1_mask=None, cdr2_mask=None, cdr3_mask=None, config_runtime=None):
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
        loss
    """
    tran_loss, angle_loss, cg_loss_all,tran_mse_loss, rmsd_loss = torch.tensor(0.0).to(predicted_tran.device),torch.tensor(0.0).to(predicted_tran.device),torch.tensor(0.0).to(predicted_tran.device),torch.tensor(0.0).to(predicted_tran.device),torch.tensor(0.0).to(predicted_tran.device)
    if config_runtime['use_cg_loss']:
        # 计算gt_backbone_atom_positions，并且筛选predicted_tran, init_model_backbone与其对应
        restore_predicted_tran2start_model_shape = torch.zeros(align_atom_positions.shape).to(predicted_tran.device).view(-1,3)
        
        # 获取start_model_backbone_mask中为 1 的位置
        indices = start_model_backbone_mask.nonzero(as_tuple=True)[0]  # shape [n], n 是 A 中 1 的数量

        # 将 predicted_tran 的值赋给 restore_predicted_tran2start_model_shape 的相应位置
        restore_predicted_tran2start_model_shape[indices] = predicted_tran
        
        refine_backbone_atom = align_atom_positions+restore_predicted_tran2start_model_shape.reshape(align_atom_positions.shape)

        # 计算refine model的cg loss.
        # 还需要mask掉未参与训练的氨基酸，即采用gt res mask和backbone mask
       
        cg_loss_all = cg_loss_function(refine_backbone_atom[cdr_mask_by_res_mask ==1], res_names_list[gt_res_mask==1][cdr_mask_by_res_mask ==1])

    if config_runtime['use_rmsd']: 
        # 计算gt_backbone_atom_positions，并且筛选predicted_tran, init_model_backbone与其对应
        restore_predicted_tran2start_model_shape = torch.zeros(align_atom_positions.shape).to(predicted_tran.device).view(-1,3)
        
        # 获取start_model_backbone_mask中为 1 的位置
        indices = start_model_backbone_mask.nonzero(as_tuple=True)[0]  # shape [n], n 是 A 中 1 的数量

        # 将 predicted_tran 的值赋给 restore_predicted_tran2start_model_shape 的相应位置
        restore_predicted_tran2start_model_shape[indices] = predicted_tran
        
        refine_backbone_atom = align_atom_positions+restore_predicted_tran2start_model_shape.reshape(align_atom_positions.shape)
        
        
        
        # #find the CA atom, where the index is 1
        # res_list_CA = torch.zeros(37).to(predicted_tran.device)
        # res_list_CA[1] = 1

        # # 获得初始化的backbone list
        # gt_res_num = gt_model.shape[0]
        # CA_list = res_list_CA.repeat(gt_res_num, 1).view(-1)
        
        # # 获取从gt atom mask中继续获得gt backbone原子的位置，其中1代表backbone原子
        # finally_CA_atom_mask = torch.logical_and(gt_atom_mask.view(-1), CA_list).int()
        
        # 计算start model的backbone rmsd
        rot_start, tran_start_, rmsd_loss, superimpose_start = superimpose_single(
            gt_model.view(-1,3), refine_backbone_atom.view(-1,3), mask= atom_mask_align2gt # mask中为1则是该原子存在，需要rmsd计算
        )



    if config_runtime['use_tran_mse_loss']:
        # 计算每个原子对的MSE
       
        mse_loss = (predicted_tran - align_tran) ** 2
        if config_runtime['use_CDR_loss'] == True:
            cdr1_loss_weight=config_runtime['cdr1_loss_weight']
            cdr2_loss_weight=config_runtime['cdr2_loss_weight']
            cdr3_loss_weight=config_runtime['cdr3_loss_weight']
            #采用cdr loss
            cdr_mask = cdr1_mask*cdr1_loss_weight + cdr2_mask*cdr2_loss_weight + cdr3_mask*cdr3_loss_weight
            cdr_mask[cdr_mask == 0] = config_runtime['no_cdr_loss_weight'] # 除了cdr区域，其他backbone的权重为no_cdr_loss_weight
            
            # 对每个原子对加权
            # 确保 weights 的形状为 [M, 1]，以便与 [M, 3] 进行广播
            cdr_mask = cdr_mask.unsqueeze(1)  # [M, 1]
            mse_loss = mse_loss * cdr_mask


        if config_runtime['tran_mse_loss_mean_or_sum'] == 'mean':
            # 对所有原子对求平均或求和
            tran_mse_loss = mse_loss.mean()  # 使用.mean() 计算每个样本的平均损失
        else:
            # 对所有原子对求平均或求和
            tran_mse_loss = mse_loss.sum()  # 使用.mean() 计算每个样本的平均损失

    else: 
        mod = magnitude_loss(v1=predicted_tran, v2=align_tran)
        angle = 1-batch_direction_error(predicted_tran, align_tran)

        if config_runtime['use_CDR_loss'] == True:
            cdr1_loss_weight=config_runtime['cdr1_loss_weight']
            cdr2_loss_weight=config_runtime['cdr2_loss_weight']
            cdr3_loss_weight=config_runtime['cdr3_loss_weight']
            #采用cdr loss
            cdr_mask = cdr1_mask*cdr1_loss_weight + cdr2_mask*cdr2_loss_weight + cdr3_mask*cdr3_loss_weight
            cdr_mask[cdr_mask == 0] = config_runtime['no_cdr_loss_weight'] # 除了cdr区域，其他backbone的权重为no_cdr_loss_weight
            mod = (mod * cdr_mask)
            angle = (angle * cdr_mask)
        

        if config_runtime['angle_loss_mean_or_sum'] == 'mean':
            angle_loss = torch.mean(angle)
        else: 
            angle_loss = torch.sum(angle)

        if config_runtime['tran_loss_mean_or_sum'] == 'mean':
            tran_loss = torch.mean(mod)
        else:
            tran_loss = torch.sum(mod)

    return tran_loss, angle_loss, cg_loss_all, tran_mse_loss, rmsd_loss



def computer_rmsd4test_only_trans(align_tran, predicted_tran, start_model_backbone_mask, init_model, start_atom_mask, gt_model, gt_res_mask, gt_atom_mask, res_names_list, start2gt_model_tran, start2gt_model_rot, cdr_mask_by_res_mask=None, cdr1_mask=None, cdr2_mask=None, cdr3_mask=None, config_runtime=None):
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
            [gt model res * 37], tensor
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

    # 计算refine model的cg loss.
    # 还需要mask掉未参与训练的氨基酸，即采用gt res mask和backbone mask
    cg_loss_all = torch.tensor(0.0).to(predicted_tran.device)
    if config_runtime['use_cg_loss']:
        # cdr_mask = cdr1_mask + cdr2_mask + cdr3_mask
        # cdr_mask = cdr_mask.view(-1, 4)
        # cdr_mask = (torch.sum(cdr_mask, dim=1) == 4).long()
        cg_loss_all = cg_loss_function(refine_backbone_atom[gt_res_mask==1][cdr_mask_by_res_mask ==1], res_names_list[gt_res_mask==1][cdr_mask_by_res_mask ==1])

    # computer the CA loss of CDR 
    CA_in_backbone = finally_CA_atom_mask[finally_backbone_atom_mask==1] #获得backbone长度的CA mask
    cdr_mask_without_weight = cdr1_mask + cdr2_mask + cdr3_mask # backbone长度cdr mask
    # 两者取并集，获得cdr区域的CA
    cdr_CA_mask = torch.logical_and(CA_in_backbone, cdr_mask_without_weight).int()
    
    rmsd_refine_CA_CDR_after = torch.sqrt(
            torch.sum(torch.sum((superimpose_refine_CA.view(-1,3)[finally_backbone_atom_mask==1] - gt_model.view(-1,3)[finally_backbone_atom_mask==1]) ** 2, dim=-1) * cdr_CA_mask)
            / (torch.sum(cdr_CA_mask) + 1e-6)
        )

    rmsd_refine_CA_CDR_before = torch.sqrt(
            torch.sum(torch.sum((superimpose_start_CA.view(-1,3)[finally_backbone_atom_mask==1] - gt_model.view(-1,3)[finally_backbone_atom_mask==1]) ** 2, dim=-1) * cdr_CA_mask)
            / (torch.sum(cdr_CA_mask) + 1e-6)
        )
    
    # 计算CDR1区域的rmsd
    cdr1_CA_backbone_mask = torch.logical_and(CA_in_backbone, cdr1_mask).int()
    
    rmsd_refine_CA_CDR1_after = torch.sqrt(
            torch.sum(torch.sum((superimpose_refine_CA.view(-1,3)[finally_backbone_atom_mask==1] - gt_model.view(-1,3)[finally_backbone_atom_mask==1]) ** 2, dim=-1) * cdr1_CA_backbone_mask)
            / (torch.sum(cdr1_CA_backbone_mask) + 1e-6)
        )

    rmsd_refine_CA_CDR1_before = torch.sqrt(
            torch.sum(torch.sum((superimpose_start_CA.view(-1,3)[finally_backbone_atom_mask==1] - gt_model.view(-1,3)[finally_backbone_atom_mask==1]) ** 2, dim=-1) * cdr1_CA_backbone_mask)
            / (torch.sum(cdr1_CA_backbone_mask) + 1e-6)
        )

    CDR1_impove_rmsd = rmsd_refine_CA_CDR1_before - rmsd_refine_CA_CDR1_after
    
    # 计算CDR2区域的rmsd
    cdr2_CA_backbone_mask = torch.logical_and(CA_in_backbone, cdr2_mask).int()
    
    rmsd_refine_CA_CDR2_after = torch.sqrt(
            torch.sum(torch.sum((superimpose_refine_CA.view(-1,3)[finally_backbone_atom_mask==1] - gt_model.view(-1,3)[finally_backbone_atom_mask==1]) ** 2, dim=-1) * cdr2_CA_backbone_mask)
            / (torch.sum(cdr2_CA_backbone_mask) + 1e-6)
        )

    rmsd_refine_CA_CDR2_before = torch.sqrt(
            torch.sum(torch.sum((superimpose_start_CA.view(-1,3)[finally_backbone_atom_mask==1] - gt_model.view(-1,3)[finally_backbone_atom_mask==1]) ** 2, dim=-1) * cdr2_CA_backbone_mask)
            / (torch.sum(cdr2_CA_backbone_mask) + 1e-6)
        )

    CDR2_impove_rmsd = rmsd_refine_CA_CDR2_before - rmsd_refine_CA_CDR2_after

    # 计算CDR3区域的rmsd
    cdr3_CA_backbone_mask = torch.logical_and(CA_in_backbone, cdr3_mask).int()
    
    rmsd_refine_CA_CDR3_after = torch.sqrt(
            torch.sum(torch.sum((superimpose_refine_CA.view(-1,3)[finally_backbone_atom_mask==1] - gt_model.view(-1,3)[finally_backbone_atom_mask==1]) ** 2, dim=-1) * cdr3_CA_backbone_mask)
            / (torch.sum(cdr3_CA_backbone_mask) + 1e-6)
        )

    rmsd_refine_CA_CDR3_before = torch.sqrt(
            torch.sum(torch.sum((superimpose_start_CA.view(-1,3)[finally_backbone_atom_mask==1] - gt_model.view(-1,3)[finally_backbone_atom_mask==1]) ** 2, dim=-1) * cdr3_CA_backbone_mask)
            / (torch.sum(cdr3_CA_backbone_mask) + 1e-6)
        )

    CDR3_impove_rmsd = rmsd_refine_CA_CDR3_before - rmsd_refine_CA_CDR3_after
    
    # 将预测的tran旋转到align的位置上
    rot_predicted_tran =(torch.matmul(start2gt_model_rot, restore_predicted_tran2start_model_shape.T).T)

    rot_predicted_tran_mask = rot_predicted_tran.reshape(init_model.shape)[gt_res_mask==1].reshape((-1,3))[finally_backbone_atom_mask==1]

    align_backbone_tran = align_tran.reshape((-1,3))[finally_backbone_atom_mask==1]

    
    tran_loss, angle_loss,tran_mse_loss = torch.tensor(0.0).to(predicted_tran.device),torch.tensor(0.0).to(predicted_tran.device),torch.tensor(0.0).to(predicted_tran.device)
    if config_runtime['use_tran_mse_loss']:
        # 计算每个原子对的MSE
        mse_loss = (rot_predicted_tran_mask - align_backbone_tran) ** 2
        if config_runtime['use_CDR_loss'] == True:
            cdr1_loss_weight=config_runtime['cdr1_loss_weight']
            cdr2_loss_weight=config_runtime['cdr2_loss_weight']
            cdr3_loss_weight=config_runtime['cdr3_loss_weight']
            #采用cdr loss
            cdr_mask = cdr1_mask*cdr1_loss_weight + cdr2_mask*cdr2_loss_weight + cdr3_mask*cdr3_loss_weight
            cdr_mask[cdr_mask == 0] = config_runtime['no_cdr_loss_weight'] # 除了cdr区域，其他backbone的权重为no_cdr_loss_weight
            
            
            # 对每个原子对加权
            # 确保 weights 的形状为 [M, 1]，以便与 [M, 3] 进行广播
            cdr_mask = cdr_mask.unsqueeze(1)  # [M, 1]
            mse_loss = mse_loss * cdr_mask


        if config_runtime['tran_mse_loss_mean_or_sum'] == 'mean':
            # 对所有原子对求平均或求和
            tran_mse_loss = mse_loss.mean()  # 使用.mean() 计算每个样本的平均损失
        else:
            # 对所有原子对求平均或求和
            tran_mse_loss = mse_loss.sum()  # 使用.mean() 计算每个样本的平均损失

    else: 
        mod = magnitude_loss(rot_predicted_tran_mask, align_backbone_tran)
        angle = 1-batch_direction_error(rot_predicted_tran_mask, align_backbone_tran)


        if config_runtime['use_CDR_loss'] == True:
            cdr1_loss_weight=config_runtime['cdr1_loss_weight']
            cdr2_loss_weight=config_runtime['cdr2_loss_weight']
            cdr3_loss_weight=config_runtime['cdr3_loss_weight']
            #采用cdr loss
            cdr_mask = cdr1_mask*cdr1_loss_weight + cdr2_mask*cdr2_loss_weight + cdr3_mask*cdr3_loss_weight
            cdr_mask[cdr_mask == 0] = config_runtime['no_cdr_loss_weight'] # 除了cdr区域，其他backbone的权重为no_cdr_loss_weight
            mod = (mod * cdr_mask)
            angle = (angle * cdr_mask)
        

        if config_runtime['angle_loss_mean_or_sum'] == 'mean':
            angle_loss = torch.mean(angle)
        else: 
            angle_loss = torch.sum(angle)

        if config_runtime['tran_loss_mean_or_sum'] == 'mean':
            tran_loss = torch.mean(mod)
        else:
            tran_loss = torch.sum(mod)
    # tran_loss = loss_mse(rot_predicted_tran.reshape((-1,3))[finally_backbone_atom_mask==1], align_tran.reshape((-1,3))[finally_backbone_atom_mask==1])

    return rmsd_refine, rmsd_start, rmsd_refine_CA, rmsd_start_CA, tran_mse_loss, tran_loss, angle_loss, cg_loss_all, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd


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

def custom_collate_fn(data_list):
        # 过滤掉 None 值
        data_list = [data for data in data_list if data is not None]
        # 将过滤后的数据组合成一个批次（可以根据具体需求进行处理）
        if not data_list:
            return None

        return data_list  #将数据进一步处理成批量形式
        
def test_run_train(args, config_runtime, config_model, config_data):
    os.environ["WANDB_IGNORE_GIT"] = "1"
    wandb_PJ_name = args.wandb_name
    experiment_name = args.experiment_name
    # local_rank = args.local_rank
    local_rank = args.rank
    yaml_name = args.yaml_path.split('/')[-1].split('.')[0]
    experiment_name = experiment_name+ "_yaml_"+yaml_name

    # local_rank = args.rank
    prefetch_factor = config_data['prefetch_factor'] 
    batch_size = config_data['trn_batch_size'] 
    epochs = config_runtime['epoch']
    lr = config_runtime['learning_rate']
    sample_num = config_data['sample_num'] 
    valid_head = config_data['valid_head'] 
    min_num_workers = config_data['min_num_workers'] 
    log_interval = config_runtime['log_interval']
    log_interval_for_test = config_runtime['log_interval_for_test']
    weight_tran =  config_runtime['weight_tran']
    weight_angle = config_runtime['weight_angle']
    weight_rmsd = config_runtime['rmsd_loss_weight']
    distmap_loss_weight = config_runtime['distmap_loss_weight']
    cg_loss_weight =config_runtime['cg_loss_weight'] 
    weight_angle_decay_rate = config_runtime['weight_angle_decay_rate']

    use_CDR_loss =config_runtime['use_CDR_loss']
    cdr1_loss_weight=config_runtime['cdr1_loss_weight']
    cdr2_loss_weight=config_runtime['cdr2_loss_weight']
    cdr3_loss_weight=config_runtime['cdr3_loss_weight']
    distmap_loss_mean_or_sum = config_runtime['distmap_loss_mean_or_sum']
    distmap_loss_use_cdr_mask = config_runtime['distmap_loss_use_cdr_mask']

    dataset_type = config_data['dataset_type']

    set_random_seed(config_runtime['seed'])
    # get the pdb fpath
    train_file_path = config_data['train_filepath'][0]['path']
    train_file_name = train_file_path.split("/")[-1].split(".")[0]
  
    test_file_path = config_data['test_filepath']
    valid_file_path = config_data['valid_filepath']
    
    num_workers = min(min_num_workers, os.cpu_count() // dist_utils.get_world_size())
    
    
    train_dataset = PDBDataset(csv_file=train_file_path, data_type='train', dataset_type=dataset_type, sample_num=sample_num, valid_head=-1)
    valid_dataset = PDBDataset(csv_file=valid_file_path, data_type='test', dataset_type=dataset_type, sample_num=-1, valid_head=-1)
    test_dataset = PDBDataset(csv_file=test_file_path, data_type='test', dataset_type=dataset_type, sample_num=-1, valid_head=-1)
    # inference_dataset = PDBDataset(csv_file=test_file_path, data_type='inference', dataset_type=dataset_type, sample_num=-1, valid_head=-1)

    
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
    
    
    
    model = EquiformerV2(**config_model).to(local_rank)
    if config_model['model_reload_path'] != False:
        model.load_state_dict(torch.load(config_model['model_reload_path']))
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    ema = EMA(model.module, decay=0.9999)

    distmap_loss_fn = nn.CrossEntropyLoss(reduction='sum' if distmap_loss_mean_or_sum == 'sum' else 'mean')
    step = 0
    test_all_pdb_CA_RMSD_improve_best = -10000
    start_epoch = 0
    #  检查本地 checkpoint 是否存在，若存在则加载
    resume_ckpt_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/'+experiment_name+".pth"
    if os.path.exists(resume_ckpt_path):
        # checkpoint = torch.load(resume_ckpt_path, map_location=)
        map_loc = {f"cuda:0": f"cuda:{local_rank}"}
        checkpoint = torch.load(resume_ckpt_path, map_location=map_loc)
        model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        if 'global_step' in checkpoint:
            step = checkpoint['global_step']
        if 'wandb_run_id' in checkpoint:
            wandb_run_id = checkpoint['wandb_run_id']
        if 'model_time_flag' in checkpoint:
            model_time_flag = checkpoint['model_time_flag']
        if 'test_all_pdb_CA_RMSD_improve_best' in checkpoint:
            test_all_pdb_CA_RMSD_improve_best = checkpoint['test_all_pdb_CA_RMSD_improve_best']
        if 'ema_state_dict' in checkpoint:
            ema.shadow = checkpoint['ema_state_dict']

        if local_rank == 0:
            # 初始化/恢复 WandB
            import wandb
            from wandb import Settings
            # my_settings = Settings(_no_git=True)
            wandb_config = wandb.init(
                project=wandb_PJ_name,
                settings=wandb.Settings(start_method='fork'),
                name=experiment_name+'_'+str(model_time_flag +'refine_' + str(local_rank)),
                id=wandb_run_id,
                dir="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/logs",
                config={
                    "architecture": "equiformer2",
                    "epochs": epochs,
                    "train_data_file": train_file_path,
                    "test_data_file": test_file_path,
                    },
                resume="allow",  # 如果 wandb_run_id 存在，则从之前的 run 恢复
            )

            print(f"Resumed from epoch={checkpoint['epoch']}, global_step={step}, "
                f"wandb_run_id={wandb_run_id}")
        else:
            os.environ["WANDB_DISABLED"] = "true"
    else:
        print("No checkpoint found. Training from scratch.")
        model_time_flag = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

        if local_rank == 0:
            import wandb
            from wandb import Settings
            # my_settings = Settings(_no_git=True)
            wandb_config = wandb.init(
                project=wandb_PJ_name, 
                settings=wandb.Settings(start_method='fork'),
                name=experiment_name+'_'+str(model_time_flag +'refine_' + str(local_rank)),
                dir="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/logs",
                config={
                "architecture": "equiformer2",
                "epochs": epochs,
                "train_data_file": train_file_path,
                "test_data_file": test_file_path,
                }
            )
        else:
            os.environ["WANDB_DISABLED"] = "true"

    
    loss_file = "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/rmsd_values_"+experiment_name+'_'+str(model_time_flag+'_refine_')+str(local_rank)+".txt"
    with open(loss_file, "w") as file:
        file.write("Step, start pdb name, rmsd_start_CA, rmsd_refine_CA, rmsd_CA_improve_rat, start rmsd, refine rmsd, improve rat, start CDR CA rmsd, refine CDR CA rmsd, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd \n")  
    
    start_time = time.time()  # 记录开始时间
    train_time_log_step = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        train_angle_loss = 0
        train_tran_loss = 0
        train_tran_mse_loss =0
        train_cg_loss = 0
        train_rmsd_loss = 0
        train_distmap_loss = 0
        train_tri_loss = 0
        # print('the train_dataloader batch num is ', len(train_dataloader))
        accumulation_steps = 8  # update weight each accumulation_steps
        optimizer.zero_grad()
        # for data in train_dataloader:
        
        for i, data in enumerate(train_dataloader):
            if data is None:
                continue
            
            model_start_time = time.time()  # model time

            data = data.to(local_rank)
            pred_distmap, predicted_atom_trans  = model(data) # align model
            if train_time_log_step < 2:
                print('model time use:', time.time() - model_start_time)

            if pred_distmap == None:
                distmap_loss = torch.tensor(0.0).to(local_rank)
                tri_loss = torch.tensor(0.0).to(local_rank)
            else:
                distmap_loss, tri_loss = distmap_loss_function(
                    pred_distmap=pred_distmap, 
                    gt_model=data.gt_atom_positions, 
                    start_model_backbone_mask=data.start_model_backbone_mask,
                    distmap_loss_fn= distmap_loss_fn,
                    cdr1_mask=data.cdr1_mask_backbone_atom, 
                    cdr2_mask=data.cdr2_mask_backbone_atom, 
                    cdr3_mask=data.cdr3_mask_backbone_atom,
                    # distmap_cdr1_mask = data.distmap_cdr1_mask,
                    # distmap_cdr2_mask = data.distmap_cdr2_mask,
                    # distmap_cdr3_mask = data.distmap_cdr3_mask,
                    use_cdr_mask=distmap_loss_use_cdr_mask
                    )
            if train_time_log_step<2:
                print('from model start to distmap loss time use:', time.time() - model_start_time)


            tran_loss, angle_loss, cg_loss_all, tran_mse_loss, rmsd_loss = loss_function_T_backbone(
                align_tran=data.align_atom_trans, 
                predicted_tran=predicted_atom_trans,
                res_names_list = data.res_names_list, 
                gt_res_mask=data.gt_res_mask, 
                gt_atom_mask=data.gt_atom_mask,
                gt_model=data.gt_atom_positions, 
                start_model_backbone_mask=data.start_model_backbone_mask, 
                init_model=data.all_atom_positions,
                align_atom_positions= data.align_atom_positions,
                atom_mask_align2gt=data.atom_mask_align2gt,
                cdr_mask_by_res_mask=data.cdr_mask_by_res_mask,
                cdr1_mask=data.cdr1_mask_backbone_atom, 
                cdr2_mask=data.cdr2_mask_backbone_atom, 
                cdr3_mask=data.cdr3_mask_backbone_atom, 
                config_runtime=config_runtime
                )

            if train_time_log_step<2:
                print('from model start to loss_function_T_backbone_start_time time use:', time.time() - model_start_time)
                
                

            loss = weight_tran * tran_loss + weight_angle * angle_loss + cg_loss_weight * cg_loss_all + distmap_loss_weight *distmap_loss + tri_loss + tran_mse_loss + rmsd_loss*weight_rmsd
            
            loss = loss / accumulation_steps  # grad norm
            loss.backward()
            # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            #     loss.backward()
            # print(prof.key_averages().table(sort_by="cuda_time_total"))
            
            if train_time_log_step<2:
                print('from model start to backward time use:', time.time() - model_start_time)

            # update weight each accumulation_steps 
            if (step ) % accumulation_steps == 0 or (step ) == len(train_dataloader):
                optimizer.step()  # update
                # update the lr
                scheduler.step()
                ema.update()
                optimizer.zero_grad()  # zero grad
            

            train_tran_loss += weight_tran * tran_loss.detach() 
            train_angle_loss += weight_angle * angle_loss.detach() 
            train_cg_loss += cg_loss_weight *cg_loss_all.detach()
            train_rmsd_loss +=  weight_rmsd * rmsd_loss.detach()
            train_tran_mse_loss += tran_mse_loss.detach()
            train_distmap_loss += distmap_loss_weight * distmap_loss.detach()
            train_tri_loss += tri_loss.detach()

            if train_time_log_step<2:
                train_time_log_step += 1
                print('A full step time (from model to here) use:', time.time()-model_start_time)

            # log the training loss each log_interval_for_test and log_interval
            
            if (step ) % log_interval_for_test == 0  and (step ) % log_interval == 0 and step != 0:
                eval_time_start = time.time()  # 记录开始时间
                ema.apply_to(model.module)
                # update the loss weight, 如果两loss之间相差一个数量级
                # if torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) > 1 and torch.abs(train_angle_loss)>torch.abs(train_tran_loss):
                #     weight_angle = weight_angle * 0.9
                # elif torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) > 1 and torch.abs(train_angle_loss) < torch.abs(train_tran_loss):
                #     weight_tran = weight_tran * 0.9
            
                # log the train info!
                avg_tran_loss = train_tran_loss.item() / (log_interval * batch_size)
                avg_angle_loss = train_angle_loss.item() / (log_interval * batch_size)
                avg_cg_loss_all_loss = train_cg_loss.item() / (log_interval * batch_size)
                avg_rmsd_loss_all_loss = train_rmsd_loss.item() / (log_interval * batch_size)
                avg_tran_mse_loss = train_tran_mse_loss.item()/ (log_interval * batch_size)
                avg_train_distmap_loss = train_distmap_loss.item() / (log_interval * batch_size)
                avg_train_tri_loss = train_tri_loss.item() / (log_interval * batch_size)
                train_angle_loss = 0 # reset 
                train_tran_loss = 0
                train_cg_loss =0
                train_rmsd_loss=0
                train_tran_mse_loss= 0
                train_distmap_loss = 0
                train_tri_loss = 0

                # log the test info!
                improve_rat_all = 0.0
                all_pdb_improve_rmsd = 0.0
                improvement_pdb_num = 0

                test_tran_mean_loss = 0.0
                test_angle_mean_loss = 0.0
                test_tran_mse_loss = 0.0
                test_cg_mean_loss = 0.0
                test_distmap_loss = 0
                test_tri_loss = 0

                valid_tran_mean_loss = 0.0
                valid_tran_mse_loss = 0.0
                valid_angle_mean_loss = 0.0
                valid_cg_loss =0
                valid_distmap_loss = 0
                valid_tri_loss = 0

                valid_ca_imporve_rmsd = 0.0
                valid_ca_rmsd_num = 0.0
                CDR_CA_improve_rmsd=0.0
                CDR1_impove_rmsd_all_test, CDR2_impove_rmsd_all_test, CDR3_impove_rmsd_all_test = 0.0, 0.0, 0.0
                CDR1_impove_rmsd_all_valid, CDR2_impove_rmsd_all_valid, CDR3_impove_rmsd_all_valid = 0.0, 0.0, 0.0
                model.eval()
                # print('To test')
                with torch.no_grad():
                    for data in tqdm(test_dataloader):
                        pred_distmap, predicted_atom_trans  = model(data.to(local_rank))

                        if pred_distmap is None:
                            distmap_loss = torch.tensor(0.0).to(local_rank)
                            tri_loss = torch.tensor(0.0).to(local_rank)
                        else:
                            distmap_loss, tri_loss = distmap_loss_function(
                                pred_distmap=pred_distmap, 
                                gt_model=data.gt_atom_positions.to(local_rank), 
                                start_model_backbone_mask=data.start_model_backbone_mask.to(local_rank),
                                distmap_loss_fn=distmap_loss_fn,
                                cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                                cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                                cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank),
                                # distmap_cdr1_mask = data.distmap_cdr1_mask,
                                # distmap_cdr2_mask = data.distmap_cdr2_mask,
                                # distmap_cdr3_mask = data.distmap_cdr3_mask,
                                use_cdr_mask=distmap_loss_use_cdr_mask
                                )

                        pdb_name = data.pdb_name[0]
                        rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, tran_mse_loss, rot_pred_tran_mse, rot_pred_angle, cg_loss_all, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd = computer_rmsd4test_only_trans(
                            align_tran=data.align_atom_trans.to(local_rank),
                            predicted_tran=predicted_atom_trans, 
                            start_model_backbone_mask= data.start_model_backbone_mask.to(local_rank),
                            init_model=data.all_atom_positions.to(local_rank), 
                            start_atom_mask=data.start_atom_mask.to(local_rank), 
                            gt_model=data.gt_atom_positions.to(local_rank), 
                            gt_res_mask=data.gt_res_mask.to(local_rank), 
                            gt_atom_mask=data.gt_atom_mask.to(local_rank),
                            res_names_list = data.res_names_list.to(local_rank),
                            start2gt_model_tran = data.start2gt_model_tran.to(local_rank),
                            start2gt_model_rot = data.start2gt_model_rot.to(local_rank),
                            cdr_mask_by_res_mask=data.cdr_mask_by_res_mask.to(local_rank),
                            cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                            cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                            cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
                            config_runtime=config_runtime
                            )
                        
                        
                        CA_improve_rmsd = (rmsd_start_CA.item() - rmsd_refine_CA.item())
                        all_pdb_improve_rmsd += CA_improve_rmsd

                        CDR_CA_improve_rmsd += (rmsd_refine_CA_CDR_before.item() - rmsd_refine_CA_CDR_after.item())
                        CDR1_impove_rmsd_all_test += CDR1_impove_rmsd.item()
                        CDR2_impove_rmsd_all_test += CDR2_impove_rmsd.item()
                        CDR3_impove_rmsd_all_test += CDR3_impove_rmsd.item()
                        backbone_improve_rmsd = (rmsd_start.item() - rmsd.item())

                        with open(loss_file, "a") as file:
                            file.write(f"Step:{step}, {pdb_name}, {rmsd_start_CA:.4f}, {rmsd_refine_CA}, {CA_improve_rmsd:.4f}, {rmsd_start:.4f}, {rmsd}, {backbone_improve_rmsd:.4f}, {rmsd_refine_CA_CDR_before:.4f}, {rmsd_refine_CA_CDR_after:.4f}, {CDR1_impove_rmsd:.4f}, {CDR2_impove_rmsd:.4f}, {CDR3_impove_rmsd:.4f}\n")
                        
                        if rmsd_start_CA.item() - rmsd_refine_CA.item() > 0: # if after refine rmsd is better than start
                            # print('start RMSD:', rmsd_start.item())
                            # print('refine RMSD:', rmsd.item())
                            # print('pred_tran_mse_loss:', rot_pred_tran_mse.item())
                            # print('pred_angle_loss:', rot_pred_angle.item())
                            # print('---------')
                            improvement_pdb_num += 1
                            improve_rat_all += CA_improve_rmsd
                        
                        test_tran_mean_loss += weight_tran * rot_pred_tran_mse.detach().item()
                        test_angle_mean_loss += weight_angle * rot_pred_angle.detach().item()
                        test_cg_mean_loss += cg_loss_weight * cg_loss_all.detach().item()
                        test_tran_mse_loss += tran_mse_loss.detach().item()
                        test_distmap_loss += distmap_loss_weight *distmap_loss.detach().item()
                        test_tri_loss += tri_loss.detach().item()

                    for data in tqdm(valid_dataloader):
                        pred_distmap, predicted_atom_trans  = model(data.to(local_rank))
                        
                        if pred_distmap is None:
                            distmap_loss = torch.tensor(0.0).to(local_rank)
                            tri_loss = torch.tensor(0.0).to(local_rank)
                        else:
                            distmap_loss, tri_loss = distmap_loss_function(
                                pred_distmap=pred_distmap, 
                                gt_model=data.gt_atom_positions.to(local_rank), 
                                start_model_backbone_mask=data.start_model_backbone_mask.to(local_rank),
                                distmap_loss_fn= distmap_loss_fn,
                                cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                                cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                                cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank),
                                # distmap_cdr1_mask = data.distmap_cdr1_mask,
                                # distmap_cdr2_mask = data.distmap_cdr2_mask,
                                # distmap_cdr3_mask = data.distmap_cdr3_mask,
                                use_cdr_mask=distmap_loss_use_cdr_mask
                                )

                        rmsd, rmsd_start, rmsd_refine_CA, rmsd_start_CA, tran_mse_loss, rot_pred_tran_mse, rot_pred_angle, cg_loss_all, rmsd_refine_CA_CDR_before, rmsd_refine_CA_CDR_after, CDR1_impove_rmsd, CDR2_impove_rmsd, CDR3_impove_rmsd = computer_rmsd4test_only_trans(
                            align_tran=data.align_atom_trans.to(local_rank),
                            predicted_tran=predicted_atom_trans, 
                            start_model_backbone_mask= data.start_model_backbone_mask.to(local_rank),
                            init_model=data.all_atom_positions.to(local_rank), 
                            start_atom_mask=data.start_atom_mask.to(local_rank), 
                            gt_model=data.gt_atom_positions.to(local_rank), 
                            gt_res_mask=data.gt_res_mask.to(local_rank), 
                            gt_atom_mask=data.gt_atom_mask.to(local_rank),
                            res_names_list = data.res_names_list.to(local_rank),
                            start2gt_model_tran = data.start2gt_model_tran.to(local_rank),
                            start2gt_model_rot = data.start2gt_model_rot.to(local_rank),
                            cdr_mask_by_res_mask=data.cdr_mask_by_res_mask.to(local_rank),
                            cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                            cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                            cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
                            config_runtime=config_runtime
                            )
                        # tran_loss, angle_loss, cg_loss_all, tran_mse_loss = loss_function_T_backbone(
                        #     align_tran=data.align_atom_trans.to(local_rank), 
                        #     predicted_tran=predicted_atom_trans,
                        #     res_names_list = data.res_names_list.to(local_rank), 
                        #     gt_res_mask=data.gt_res_mask.to(local_rank), 
                        #     start_model_backbone_mask=data.start_model_backbone_mask.to(local_rank), 
                        #     init_model=data.all_atom_positions.to(local_rank),
                        #     cdr1_mask=data.cdr1_mask_backbone_atom.to(local_rank), 
                        #     cdr2_mask=data.cdr2_mask_backbone_atom.to(local_rank), 
                        #     cdr3_mask=data.cdr3_mask_backbone_atom.to(local_rank), 
                        #     config_runtime=config_runtime
                        #     )
                        CDR1_impove_rmsd_all_valid += CDR1_impove_rmsd.item()
                        CDR2_impove_rmsd_all_valid += CDR2_impove_rmsd.item()
                        CDR3_impove_rmsd_all_valid += CDR3_impove_rmsd.item()

                        valid_tran_mean_loss += weight_tran*tran_loss.detach().item()
                        valid_tran_mse_loss += tran_mse_loss.detach().item()
                        valid_angle_mean_loss += weight_angle*angle_loss.detach().item()
                        valid_cg_loss += cg_loss_weight * cg_loss_all.detach().item()
                        valid_distmap_loss += distmap_loss_weight *distmap_loss.detach().item()
                        valid_tri_loss += tri_loss.detach().item()

                        valid_ca_imporve_rmsd += (rmsd_start_CA.item() - rmsd_refine_CA.item())
                        if rmsd_start_CA.item() - rmsd_refine_CA.item() > 0: # if after refine rmsd is better than start
                            valid_ca_rmsd_num += 1
                    # log metrics to wandb
                    metrics = {
                        "test_CA_improvement_mean_improvement": improve_rat_all / (improvement_pdb_num+1e-8),
                        "test_CA_improvement_pdb_num": improvement_pdb_num,
                        "valid_all_pdb_CA_improvement_mean_RMSD": valid_ca_imporve_rmsd / len(valid_dataloader),
                        "valid_CA_improvement_pdb_num": valid_ca_rmsd_num,
                        "test_all_pdb_CA_RMSD_improve_": all_pdb_improve_rmsd/len(test_dataloader),
                        "test_all_pdb_CDR_CA_RMSD_improve_": CDR_CA_improve_rmsd/len(test_dataloader),
                        "CDR1_impove_rmsd_all_test": CDR1_impove_rmsd_all_test/len(test_dataloader),
                        "CDR2_impove_rmsd_all_test": CDR2_impove_rmsd_all_test/len(test_dataloader),
                        "CDR3_impove_rmsd_all_test": CDR3_impove_rmsd_all_test/len(test_dataloader),
                        "test_tran_mean_loss": test_tran_mean_loss / len(test_dataloader),
                        "test_tran_mse_loss": test_tran_mse_loss/ len(test_dataloader),
                        "test_angle_mean_loss": test_angle_mean_loss / len(test_dataloader),
                        "test_cg_mean_loss": test_cg_mean_loss / len(test_dataloader),
                        "test_distmap_loss": test_distmap_loss/ len(test_dataloader),
                        "test_tri_loss": test_tri_loss/ len(test_dataloader),
                        "CDR1_impove_rmsd_all_valid": CDR1_impove_rmsd_all_valid/len(valid_dataloader),
                        "CDR2_impove_rmsd_all_valid": CDR2_impove_rmsd_all_valid/len(valid_dataloader),
                        "CDR3_impove_rmsd_all_valid": CDR3_impove_rmsd_all_valid/len(valid_dataloader),
                        "valid_tran_mean_loss": valid_tran_mean_loss / len(valid_dataloader),
                        "valid_tran_mse_loss": valid_tran_mse_loss/ len(valid_dataloader),
                        "valid_angle_mean_loss": valid_angle_mean_loss / len(valid_dataloader),
                        "valid_cg_loss":valid_cg_loss/ len(valid_dataloader),
                        "valid_distmap_loss": valid_distmap_loss/ len(valid_dataloader),
                        "valid_tri_loss": valid_tri_loss/len(valid_dataloader),
                        "avg_tran_loss_log_interval": avg_tran_loss,
                        "avg_angle_loss_log_interval": avg_angle_loss,
                        "avg_backbone_cg_loss":avg_cg_loss_all_loss,
                        "avg_rmsd_loss":avg_rmsd_loss_all_loss,
                        "avg_tran_mse_loss":avg_tran_mse_loss,
                        "avg_train_distmap_loss":avg_train_distmap_loss,
                        "avg_train_tri_loss":avg_train_tri_loss,
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
                            if name == "test_CA_improvement_pdb_num" or name == "valid_CA_improvement_pdb_num":
                                # 对 CA_improvement_pdb_num 进行累加
                                averaged_metrics[name] = sum(values)
                            else:
                                # 对其他指标取平均值
                                print('------------')
                                print(name)
                                averaged_metrics[name] = sum(values) / len(values)

                        # 将所有平均值打包成一个字典，并上传到 W&B
                        averaged_metrics["step"] = step  # 将 step 加入字典
                        wandb_config.log(averaged_metrics)  # 一次性上传所有指标
                    
                        if averaged_metrics['valid_all_pdb_CA_improvement_mean_RMSD'] > test_all_pdb_CA_RMSD_improve_best:
                            test_all_pdb_CA_RMSD_improve_best = averaged_metrics['valid_all_pdb_CA_improvement_mean_RMSD']
                            torch.save(model.module.state_dict(), "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/"+experiment_name+'_'+str(model_time_flag)+'_Step_'+str(step)+'_validation_data_improve_rmsd_'+str(averaged_metrics['valid_all_pdb_CA_improvement_mean_RMSD'])+'_imporve_num_'+str(averaged_metrics['valid_CA_improvement_pdb_num'])+".ckpt")

                        # === 保存 checkpoint ===
                        ckpt_dict = {
                            'epoch': epoch,
                            'global_step': step,
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }
                        if scheduler:
                            ckpt_dict['scheduler_state_dict'] = scheduler.state_dict()

                        # 保存wandb run信息（id, name），以实现断点后继续写入同一个 run
                        ckpt_dict['ema_state_dict'] = ema.shadow
                        ckpt_dict['wandb_run_id'] = wandb.run.id
                        ckpt_dict['model_time_flag'] = model_time_flag
                        ckpt_dict['test_all_pdb_CA_RMSD_improve_best'] = test_all_pdb_CA_RMSD_improve_best

                        torch.save(ckpt_dict, "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/model/ckpt/"+experiment_name+".pth")
                #切换回train
                print('eval time use:', time.time()- eval_time_start)
                model.train()
            elif (step ) % log_interval == 0 : # log the data each log_interval
                # update the loss weight
                # if torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) > 1 and torch.abs(train_angle_loss) > torch.abs(train_tran_loss):
                #     weight_angle = weight_angle * 0.9
                # elif torch.abs(torch.log10(torch.abs(train_tran_loss)) - torch.log10(torch.abs(train_angle_loss))) > 1 and torch.abs(train_angle_loss) < torch.abs(train_tran_loss):
                #     weight_tran = weight_tran * 0.9
            
                # print('log the train info!')
                # log the time 
                print('step:',step,' time use:', time.time() - start_time)

                avg_tran_loss = train_tran_loss.item() / (log_interval * batch_size)
                avg_angle_loss = train_angle_loss.item() / (log_interval * batch_size)
                avg_cg_loss_all_loss = train_cg_loss.item() / (log_interval * batch_size)
                avg_rmsd_loss_all_loss = train_rmsd_loss.item() / (log_interval * batch_size)
                avg_tran_mse_loss = train_tran_mse_loss.item()/ (log_interval * batch_size)
                avg_train_distmap_loss = train_distmap_loss.item() / (log_interval * batch_size)
                avg_train_tri_loss = train_tri_loss.item() / (log_interval * batch_size)
                train_angle_loss = 0 # reset 
                train_tran_loss = 0
                train_rmsd_loss = 0
                train_tran_mse_loss = 0
                train_cg_loss =0
                train_distmap_loss = 0
                train_tri_loss = 0

                metrics = {
                        "avg_tran_loss_log_interval": avg_tran_loss,
                        "avg_angle_loss_log_interval": avg_angle_loss,
                        "avg_backbone_cg_loss":avg_cg_loss_all_loss,
                        "avg_rmsd_loss":avg_rmsd_loss_all_loss,
                        "avg_tran_mse_loss":avg_tran_mse_loss,
                        "avg_train_distmap_loss":avg_train_distmap_loss,
                        "avg_train_tri_loss":avg_train_tri_loss,
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
                    wandb_config.log(averaged_metrics)  # 一次性上传所有指标
                      

            step += 1



    
# for i, data in enumerate(train_dataloader):
#     if data is None:
#         continue
#     optimizer.zero_grad()
#     # 开始FM
#     batch_size = int(data.batch.max()) + 1
#     t = torch.rand(batch_size).type_as(data.pos)
#     epsilon = torch.randn_like(data.pos)
#     # 扩展t以匹配节点数量
#     t = t[data.batch]  # [num_nodes]
#     t = t.unsqueeze(-1)   # [num_nodes, 1]
    
#     # 对位置进行插值
#     mu_t = data.pos + (t * data.align_atom_trans)

#     sigma_t = 0
    
#     # 扩展sigma_t以匹配节点维度
#     sigma_t = torch.full_like(mu_t, sigma_t)

#     # 复制x0_data并只更新位置
#     xt_data = data.clone()
#     xt_data.pos = mu_t + sigma_t * epsilon

#     ut = xt_data.align_atom_trans.to(local_rank)
    
    
#     pred_distmap, predicted_atom_trans  = model(xt_data.to(local_rank), t.to(local_rank)) # align model
    

#     for data in tqdm(test_dataloader):
#                         # 先进行FM操作
#                         steps = 100

#                         dt = 1.0 / steps
                            
#                         xt_data = data.clone()
#                         batch_size = int(data.ptr[-1])
                        
#                         for i in range(steps):
#                             t = torch.ones(batch_size, 1) * i * dt
                            
#                             # 改进的Euler方法，保持二阶精度
#                             pred_distmap, v = model(xt_data.to(local_rank), t.to(local_rank))
#                             xt_data.pos += 0.5 * v * dt
                            
#                             t_mid = t + 0.5 * dt
#                             pred_distmap, v_mid = model(xt_data.to(local_rank), t_mid.to(local_rank))
                            
#                             # 更新保持等变性
#                             xt_data.pos += v_mid * dt

#                         #获得xt之后，再进行验证  
#                         predicted_atom_trans = xt_data.pos - data.pos
