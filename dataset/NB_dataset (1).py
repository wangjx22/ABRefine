
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

def custom_collate_fn(data_list):
    # 过滤掉 None 值
    data_list = [data for data in data_list if data is not None]
    # 将过滤后的数据组合成一个批次（可以根据具体需求进行处理）
    if not data_list:
        return None

    return data_list  #将数据进一步处理成批量形式

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
