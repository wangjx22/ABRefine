"""Code."""
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
from utils.rmsd import superimpose as si
from utils.protein import get_seq_info
from utils.constants import cg_constants
from utils.logger import Logger
from utils.opt_utils import superimpose_single, masked_differentiable_rmsd
from dataset import numbering
import torch
import math
from scipy.spatial.transform import Rotation as R
logger = Logger.logger
import matplotlib.pyplot as plt
FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]
PICO_TO_ANGSTROM = 0.01
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
from Bio.PDB import PDBParser, PDBIO, Atom
import time
import os

@dataclasses.dataclass(frozen=True)
class Protein:
    """Define Class Protein."""

    """Protein structure representation."""
    rmsd: np.ndarray # rmsd between start model and gt model
    atom_positions: np.ndarray # start model pos
    align_pos: np.ndarray # align model pos
    start2gt_model_tran: np.ndarray # tran of start model to gt 
    start2gt_model_rot: np.ndarray # rot of start model to gt 
    atom_trans_align2gt : np.ndarray # atom trans of align to gt
    trans_target: np.ndarray # cg trans of align to gt 
    rot_target: np.ndarray # cg rot of align to gt 
    rot_m4_target: np.ndarray # cg m4 of align to gt 
    gt_res_mask: np.ndarray # res mask of full_seq_AMR
    gt_atom_positions: np.ndarray # gt model pos
    cdr1_mask: np.ndarray
    cdr2_mask: np.ndarray
    cdr3_mask: np.ndarray
    atom2cgids: np.ndarray
    aatype: np.ndarray
    atom_mask: np.ndarray # atom mask of start model
    gt_atom_mask: np.ndarray #atom mask of gt model
    residue_index: np.ndarray
    b_factors: np.ndarray
    chain_index: np.ndarray
    atom_names: np.ndarray
    residue_names: np.ndarray
    elements: np.ndarray
    ele2nums: np.ndarray
    # hetfields: np.ndarray
    resids: np.ndarray
    # icodes: np.ndarray
    chains: list
    remark: Optional[str] = None
    parents: Optional[Sequence[str]] = None
    parents_chain_index: Optional[Sequence[int]] = None
    resolution: any = None

    def __post_init__(self):
        """Run __post_init__ method."""
        # code.
        """
    __post_init__:
    Args:
        self : self
    Returns:
    """
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains because these cannot be written to PDB format'
            )
import io
def update_pdb_with_new_coords(start_pdb_filepath, new_coords, output_pdb):
    """
    give orignal pdb and update pos, output the new pdb. 把对应backbone更新,其他的都删掉.无需mask

    input:
    - start_pdb_filepath: START model pdb file, res length == N 
    - new_coords: update_pos,shape is [N, 37, 3]
    - output_pdb: the file path of new pdb
    """
    with open(start_pdb_filepath, 'r') as f:
        pdb_str = f.read()

    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)

    backbone_atom = ['CA', 'N', 'C', 'O']
    ca_only = False
    accept_res_cnt = 0
    for model in structure:
        for chain in model:
            res_num = 0
            for res_index, res in enumerate(chain):
                # if resmask[res_index] != 1: # if this res is miss 
                #     continue
                # has_ca_flag = False
                # for atom in res:
                #     if atom.name == 'CA':
                #         has_ca_flag = True
                #         break

                if res.id[0] != ' ':
                    continue
                atoms_to_remove = []
                for atom_index, atom in enumerate(res):
                    if atom.name not in residue_constants.atom_types:
                        continue
                    
                    # 如果是backbone上的原子，则更新
                    if atom.name in backbone_atom:
                        atom_order = residue_constants.atom_order[atom.name]
                        # 更新该原子的坐标
                        new_coord = new_coords[res_num, atom_order]
                        atom.set_coord(new_coord)
                    else:
                        # 记录需要移除的原子
                        atoms_to_remove.append(atom)
                for atom in atoms_to_remove:
                    res.detach_child(atom.get_id())
                res_num += 1
    # 保存更新后的PDB文件
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)

    #将af_pdb_filepath中的/换成_，然后作为新的名字。output_pdb只是提供路径。
    save_path = output_pdb+start_pdb_filepath.replace("/", "_").split('.p')[0] +'_refine.pdb'

    # 获取目录路径
    output_dir = os.path.dirname(save_path)
    # 如果目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    pdb_io.save(save_path)
    return save_path

def read_pdb(input_chain):
    res_list = list()  # a list of residue instance
    res1_list = list()  # a list of type1 amino acid
    for i, res in enumerate(input_chain):
        if res.id[0] != " ":  # skip HETATM
            continue
        # res_idx = res.id[1]
        # icode = res.id[2].strip()
        # idx = f"{res_idx}{icode}"
        res1 = residue_constants.restype_3to1.get(res.resname.strip(), "X")
        res_list.append(res)
        res1_list.append(res1)

    pdb_seq = "".join(res1_list)
    # res_list 包含链中的所有残基实例，pdb_seq 是由单字母代码组成的该链的氨基酸序列。
    return res_list, pdb_seq



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


def from_pdb_string(
    pdb_str: str,
    gt_pdb_str: str,
    chain_id: Optional[str] = None,
    return_id2seq: bool = False,
    ca_only=False,
    full_seq_AMR=None,
    data_type='train',
    dataset_type=None
) -> Protein:
    """Run from_pdb_string method."""
    # code.
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If chain_id is specified (e.g. A), then only that chain is
      parsed. Else, all chains are parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    # pdb_fh = io.StringIO(pdb_str)
    # parser = PDBParser(QUIET=True)
    # structure = parser.get_structure('none', pdb_fh)
    # resolution = structure.header['resolution']
    # if resolution is None:
    #     resolution = 0.0
    # models = list(structure.get_models())
    # model = models[0]


    if data_type == 'inference':
        model, resolution, structure = get_model_from_str(pdb_str)

        chain_id_set = set()
        
        # update for checking the full_seq_AMR to solve the cut_fv issue.
        
        # No full_seq_AMR given. Ignore the missing residue and cut fv issue
        # All residues in the PDB file are used.
        start_idx = 0
        end_idx = -1

        pdb_seq = get_pdb_seq_by_CA(model, chain_id=None)
        

        cdr1_mask=[0]
        cdr2_mask=[0]
        cdr3_mask=[0]
        # judge dataset_type is NB?
        if dataset_type == 'NB':
            # computer the CDR index
            str_fr1, str_fr2, str_fr3, str_fr4, str_cdr1, str_cdr2, str_cdr3 = numbering.make_numbering_by_api(pdb_seq)
            cdr1_index_start, cdr1_index_end = numbering.find_substring_indices_regex(pdb_seq, str_cdr1)
            cdr1_mask = [0] * len(pdb_seq)
            cdr1_mask[cdr1_index_start:cdr1_index_end+1] = [1] * (cdr1_index_end-cdr1_index_start+1)

            cdr2_index_start, cdr2_index_end = numbering.find_substring_indices_regex(pdb_seq, str_cdr2)
            cdr2_mask = [0] * len(pdb_seq)
            cdr2_mask[cdr2_index_start:cdr2_index_end+1] = [1] * (cdr2_index_end-cdr2_index_start+1)
            
            cdr3_index_start, cdr3_index_end = numbering.find_substring_indices_regex(pdb_seq, str_cdr3)
            cdr3_mask = [0] * len(pdb_seq)
            cdr3_mask[cdr3_index_start:cdr3_index_end+1] = [1] * (cdr3_index_end-cdr3_index_start+1)
        

        # is_add_list = [0] * len(pdb_seq)

        # update for checking number of chains of the input pdb.
        chain_cnt = 0
        for _ in model:
            chain_cnt += 1

        if chain_id is None and chain_cnt > 1:
            logger.warning(f"Multiple chains detected in the PDB file. Please specify chain_id.")
            logger.warning(f"Read multiple chains as default.")

        
        # get the Start model information
        atom_positions = []
        atom2cgids = []
        aatype = []
        atom_mask = []
        residue_index = []
        chain_ids = []
        b_factors = []
        id2seq = {}
        atom_names = []
        residue_names = []
        elements = []
        ele2nums = []
        hetfields = []
        resids = []
        icodes = []
        chains = []
        # considering the residue missing CA, we should ignore these residues since they are not included in gt_fv_seq.
        accept_res_cnt = 0
      
        for chain in model:
            cur_chain_aatype = []
            
            cur_idx = -1
            for res in chain:
                has_ca_flag = False
                for atom in res:
                    if atom.name == 'CA':
                        has_ca_flag = True
                        break

                if has_ca_flag: 
                    # only when the residue has ca , this residue will be counted to cur_idx.
                    resname = residue_constants.restype_3to1.get(res.resname, 'X')
                    if resname != 'X':
                        cur_idx += 1
                    else:
                        continue
                else:
                    continue
                # has_ca_flag = False
                # for atom in res:
                #     if atom.name == 'CA':
                #         has_ca_flag = True
                #         break

                # if has_ca_flag:
                #     # only when the residue has ca, this residue will be counted to cur_idx.
                #     cur_idx += 1

                # if a residue has no CA, this residue should not be count into cur_idx.
                # but the information of other heavy atoms of this residue can be used.
                # so we don't skip the case that has_ca_flag == False

                # if the first residue has no CA, then cur_idx = -1, if start_idx = 0, then it will skip this first residue.
                # and if the residue has CA but this residue is not in the range of start_idx and end_idx (not a fv res), then it will skip this residue.
                if cur_idx < start_idx:
                    continue
                # if this residue exceed the range of end_idx, then skip this residue
                # if end_idx < 0, which means full_seq_AMR is not given, so we won't skip.
                elif (cur_idx >= end_idx) and (end_idx > 0):
                    continue
                # if cur_idx is in the range of [start_idx, end_idx], but this residue has no CA, it is actually should be treat as missing residue.
                # We can preserve the information of this residue, but it should not take this residue into account when counting accepted residues.
                
                
                res_shortname = residue_constants.restype_3to1.get(res.resname, 'X'
                                                                )
                
                                
                restype_idx = residue_constants.restype_order.get(res_shortname,
                                                                residue_constants.restype_num)
                atom_type_num = residue_constants.atom_type_num
                pos = np.zeros((atom_type_num, 3))
                atom2cgid = np.zeros((atom_type_num, 4)) # max cg num is 4 in a res 
                mask = np.zeros((atom_type_num,))
                res_b_factors = np.zeros((atom_type_num,))
                atom_name = np.empty((atom_type_num,), dtype=object)
                residue_name = np.empty((atom_type_num,), dtype=object)
                element = np.empty((atom_type_num,), dtype=object)
                ele2num = np.empty((atom_type_num,), dtype=object)
                hetfield = np.empty((atom_type_num,), dtype=object)
                resid = np.zeros((atom_type_num,))
                icode = np.empty((atom_type_num,), dtype=object)

                # print(cur_idx)
                for atom in res:
                    if atom.name not in residue_constants.atom_types:
                        continue
                    if ca_only:
                        if atom.name != 'CA':
                            continue

                    # get cg id from residue and atom
                    cg_id_type = [0,0,0,0]
                    # 暂时不需要用到cg
                    # cg_list = cg_constants.cg_dict[res.resname]
                    # for i in range(len(cg_list)):
                    #     if atom.name in cg_list[i]:
                    #         cg_id_type[i] = 1 

                    # residue_constants.atom_order[atom.name] is atom_id
                    atom2cgid[residue_constants.atom_order[atom.name]] = cg_id_type
                    pos[residue_constants.atom_order[atom.name]] = atom.coord
                    mask[residue_constants.atom_order[atom.name]] = 1
                    res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
                    atom_name[residue_constants.atom_order[atom.name]] = atom.name
                    residue_name[residue_constants.atom_order[atom.name]] = res.resname
                    element[residue_constants.atom_order[atom.name]] = atom.element
                    ele2num[residue_constants.atom_order[atom.name]] = residue_constants.ele2num[atom.element]
                    # hetfield[residue_constants.atom_order[atom.name]] = het_flag
                    resid[residue_constants.atom_order[atom.name]] = len(resids) + 1
                    # icode[residue_constants.atom_order[atom.name]] = ic

                if np.sum(mask) < 0.5:
                    continue
                aatype.append(restype_idx)
                cur_chain_aatype.append(restype_idx)
                atom_positions.append(pos)
                atom2cgids.append(atom2cgid)
                atom_mask.append(mask)
                residue_index.append(res.id[1])
                chain_ids.append(chain.id)
                b_factors.append(res_b_factors)
                atom_names.append(atom_name)
                residue_names.append(residue_name)
                elements.append(element)
                ele2nums.append(ele2num)
                # hetfields.append(hetfield)
                resids.append(resid)
                # icodes.append(icode)

            id2seq[chain.id] = cur_chain_aatype
        

        
        unique_chain_ids = np.unique(chain_ids)
        chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
        chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

        # if we only use CA atoms, then the accept_res_cnt should be the same as the number of nodes.
        # if ca_only:
        #     assert accept_res_cnt == len(residue_names), "accept_res_cnt should be the same as the number of nodes. Check what's wrong."

        
        return Protein(
            rmsd = np.array(0),
            start2gt_model_tran=np.array(0), 
            start2gt_model_rot=np.array(0),
            trans_target=np.array(0),
            atom_trans_align2gt=np.array(0),
            rot_target=np.array(0),
            rot_m4_target=np.array(0),
            gt_res_mask=np.array(0),
            cdr1_mask=np.array(cdr1_mask),
            cdr2_mask=np.array(cdr2_mask),
            cdr3_mask=np.array(cdr3_mask),
            atom_positions=np.array(atom_positions),
            align_pos = np.array(0),
            gt_atom_positions=np.array(0),
            atom2cgids = np.array(atom2cgids),
            atom_mask=np.array(atom_mask),
            gt_atom_mask=np.array(0),
            aatype=np.array(aatype),
            chain_index=chain_index,
            residue_index=np.array(residue_index),
            b_factors=np.array(b_factors),
            resolution=resolution,
            atom_names=np.array(atom_names),
            residue_names=np.array(residue_names),
            elements=np.array(elements),
            ele2nums=np.array(ele2nums),
            # hetfields=np.array(hetfields),
            resids=np.array(resids).astype(int),
            # icodes=np.array(icodes),
            chains=chains,
        )
    # train and test data 
    else:
        # get model function wastes a large time
        model, resolution, structure = get_model_from_str(pdb_str)
        gt_model, _, _structure = get_model_from_str(gt_pdb_str)

        chain_id_set = set()
        if chain_id is not None:
            if ',' in chain_id:
                chain_id = chain_id.split(',')
            chain_id_set = set(chain_id)

        # update for checking the full_seq_AMR to solve the cut_fv issue.
        
        pdb_seq = get_pdb_seq_by_CA(model, chain_id=None) # start model dont need the chain id
        # print('start model length:', len(pdb_seq))
        

        cdr1_mask=[0]
        cdr2_mask=[0]
        cdr3_mask=[0]
        # judge dataset_type is NB?
        if dataset_type == 'NB':
            # computer the CDR index
            str_fr1, str_fr2, str_fr3, str_fr4, str_cdr1, str_cdr2, str_cdr3 = numbering.make_numbering_by_api(pdb_seq)
            cdr1_index_start, cdr1_index_end = numbering.find_substring_indices_regex(pdb_seq, str_cdr1)
            cdr1_mask = [0] * len(pdb_seq)
            cdr1_mask[cdr1_index_start:cdr1_index_end+1] = [1] * (cdr1_index_end-cdr1_index_start+1)

            cdr2_index_start, cdr2_index_end = numbering.find_substring_indices_regex(pdb_seq, str_cdr2)
            cdr2_mask = [0] * len(pdb_seq)
            cdr2_mask[cdr2_index_start:cdr2_index_end+1] = [1] * (cdr2_index_end-cdr2_index_start+1)
            
            cdr3_index_start, cdr3_index_end = numbering.find_substring_indices_regex(pdb_seq, str_cdr3)
            cdr3_mask = [0] * len(pdb_seq)
            cdr3_mask[cdr3_index_start:cdr3_index_end+1] = [1] * (cdr3_index_end-cdr3_index_start+1)
        

        if full_seq_AMR is not None:
            # print('full_seq_AMR:', len(full_seq_AMR))
            # full_seq_AMR is given, need to check the AMR and cut_fv issue.
            
            cleaned_seq, is_add_list = get_seq_info(full_seq_AMR) # cleaned_seq is full seq include missing res

            # print('cleaned_seq length:', len(cleaned_seq))
            if pdb_seq != cleaned_seq: # af length != full AMR length
                print('the res of af model is not equ the cleaned_seq of full_seq_AMR')
                return False
            assert len(cleaned_seq) == len(is_add_list), "The length of cleaned_seq and is_add_list should be the same"
            
            gt_fv_seq = [e for e, c in zip(cleaned_seq, is_add_list) if c == 0]
            gt_fv_seq = "".join(gt_fv_seq)
            # print('gt_fv_seq:', len(gt_fv_seq))
            pdb_seq_gt = get_pdb_seq_by_CA(gt_model, chain_id)
            # print('pdb_seq_gt:', len(pdb_seq_gt))
            # if pdb_seq_gt != gt_fv_seq: # gt_fv_seq为pdb_seq_gt的子集
            #     print('the res of gt_model is not equ the full_seq_AMR, next to find gt_fv_seq in pdb_seq_gt!')
            
            # pdb_seq, the aa sequence from the given pdb file
            # cleaned_seq, the fv sequence, full_seq_AMR, without brackets.
            # gt_fv_seq, the subsequence of the full_seq_AMR, which the ground truth pdb contains.
            # considering the missing residue issue, the gt_fv_seq may not be a contiguous subsequence of the full_seq_AMR
            # but it must be a contiguous subsequence of pdb_seq! Otherwise the input data is not correct.

            if pdb_seq_gt.strip() == gt_fv_seq.strip():
                # sequence in pdb is totally the same with full_seq_AMR, i.e. the fv sequence.
                # indicates no missing residue, no cut fv issue.
                # Probably decoys predicted by upstream single models.
                # start from 0, end by len(pdb_seq)
                # and as the pdb is not gt, so the gt_fv_seq may not match the pdb_seq.
                start_idx = 0
                end_idx = len(pdb_seq_gt)
            else:
                # with missing residue, or cut fv issue.
                start_idx = pdb_seq_gt.find(gt_fv_seq)      # gt_fv_seq should be a contiguous subsequence of pdb_seq! get the start idx.
                assert start_idx != -1, "gt_fv_seq should be a contiguous subsequence of pdb_seq!"   # -1 indicates not found.
                end_idx = start_idx + len(gt_fv_seq)
                assert end_idx <= len(pdb_seq_gt), "end_idx should not exceed the length of pdb_seq."

        else:
            # No full_seq_AMR given. Ignore the missing residue and cut fv issue
            # All residues in the PDB file are used.
            start_idx = 0
            end_idx = -1

            pdb_seq = get_pdb_seq_by_CA(model, chain_id=None)
            gt_pdb_seq = get_pdb_seq_by_CA(gt_model, chain_id)
            if pdb_seq != gt_pdb_seq:
                print('the res of af model is not equ the gt model without full_seq_AMR')
                return False
            is_add_list = [0] * len(pdb_seq)

        gt_res_mask = [1- isad for isad in is_add_list]
        # update for checking number of chains of the input pdb.
        chain_cnt = 0
        for _ in model:
            chain_cnt += 1

        if chain_id is None and chain_cnt > 1:
            logger.warning(f"Multiple chains detected in the PDB file. Please specify chain_id.")
            logger.warning(f"Read multiple chains as default.")

        # get the gt model pos and mask
        gt_atom_positions = []
        gt_atom_mask = []
        gt_cg_mask = []
        atom_type_num = residue_constants.atom_type_num
        accept_res_cnt = 0
        res_num_flag = 0
        # print('start_idx:', start_idx)
        # print('end_idx:', end_idx)
        for chain in gt_model:
            cur_chain_aatype = []
            if chain_id is not None and chain.id not in chain_id_set:
                continue
            cur_idx = -1

            for res in chain:
                # res_hete = False
                # het_flag, resseq, ic = res.id
                # if ic == ' ':
                #     ic = ''
                # if het_flag == ' ':
                #     het_flag = 'A'
                # if res.id[0] != ' ':
                #     print(res.id)
                #     # continue
                #     res_hete = True

                has_ca_flag = False
                for atom in res:
                    if atom.name == 'CA':
                        has_ca_flag = True
                        break

                if has_ca_flag: 
                    # only when the residue has ca , this residue will be counted to cur_idx.
                    resname = residue_constants.restype_3to1.get(res.resname, 'X')
                    if resname != 'X':
                        cur_idx += 1
                    else:
                        continue
                else:
                    continue
                # if res_hete: # is heteroatom
                #     continue
            

                # if a residue has no CA, this residue should not be count into cur_idx.
                # but the information of other heavy atoms of this residue can be used.
                # so we don't skip the case that has_ca_flag == False

                # if the first residue has no CA, then cur_idx = -1, if start_idx = 0, then it will skip this first residue.
                # and if the residue has CA but this residue is not in the range of start_idx and end_idx (not a fv res), then it will skip this residue.
                if cur_idx < start_idx:
                    continue
                # if this residue exceed the range of end_idx, then skip this residue
                # if end_idx < 0, which means full_seq_AMR is not given, so we won't skip.
                if (cur_idx >= end_idx) and (end_idx > 0):
                    continue
                # if cur_idx is in the range of [start_idx, end_idx], but this residue has no CA, it is actually should be treat as missing residue.
                # We can preserve the information of this residue, but it should not take this residue into account when counting accepted residues.
                # if res_hete_flag: 
                #     accept_res_cnt += 1
                    

                
                pos = np.zeros((atom_type_num, 3))
                mask = np.zeros((atom_type_num,))
                cg_mask = np.zeros((4,))

                # cg_id_type = [0] * 4
                    
                # cg_list = cg_constants.cg_dict[res.resname]
                # for i in range(len(cg_list)):
                #     if atom.name in cg_list[i]:
                #         res_id = residue_constants.restype_order[residue_constants.restype_3to1[res.resname]]
                #         cg_id = cg_constants.cg2id[(res_id, i)] # (残积id, 该残积上的第i个cg): cg id
                #         cg_id_type[i] = 1 
            

                for atom in res:
                    if atom.name not in residue_constants.atom_types:
                        continue
                    if ca_only:
                        if atom.name != 'CA':
                            continue

                    # residue_constants.atom_order[atom.name] is atom_id
                    
                    
                    pos[residue_constants.atom_order[atom.name]] = atom.coord
                    mask[residue_constants.atom_order[atom.name]] = 1
                    
                
                gt_atom_positions.append(pos)
                gt_atom_mask.append(mask)
                res_num_flag += 1
                if res_num_flag == sum(gt_res_mask):
                    break
        

        # get the Start model information
        atom_positions = []
        atom2cgids = []
        aatype = []
        atom_mask = []
        residue_index = []
        chain_ids = []
        b_factors = []
        id2seq = {}
        atom_names = []
        residue_names = []
        elements = []
        ele2nums = []
        hetfields = []
        resids = []
        icodes = []
        chains = []
        # considering the residue missing CA, we should ignore these residues since they are not included in gt_fv_seq.
        accept_res_cnt = 0
        # start model确定都是从头开始的，所以下面两个位置参数重设
        start_idx = 0
        end_idx = len(pdb_seq)
        for chain in model:
            cur_chain_aatype = []
            # if chain_id is not None and chain.id not in chain_id_set:
            #     continue
            # chains.append(chain.id)
            cur_idx = -1
            for res in chain:
                has_ca_flag = False
                for atom in res:
                    if atom.name == 'CA':
                        has_ca_flag = True
                        break

                if has_ca_flag: 
                    # only when the residue has ca , this residue will be counted to cur_idx.
                    resname = residue_constants.restype_3to1.get(res.resname, 'X')
                    if resname != 'X':
                        cur_idx += 1
                    else:
                        continue
                else:
                    continue
                # has_ca_flag = False
                # for atom in res:
                #     if atom.name == 'CA':
                #         has_ca_flag = True
                #         break

                # if has_ca_flag:
                #     # only when the residue has ca, this residue will be counted to cur_idx.
                #     cur_idx += 1

                # if a residue has no CA, this residue should not be count into cur_idx.
                # but the information of other heavy atoms of this residue can be used.
                # so we don't skip the case that has_ca_flag == False

                # if the first residue has no CA, then cur_idx = -1, if start_idx = 0, then it will skip this first residue.
                # and if the residue has CA but this residue is not in the range of start_idx and end_idx (not a fv res), then it will skip this residue.
                if cur_idx < start_idx:
                    continue
                # if this residue exceed the range of end_idx, then skip this residue
                # if end_idx < 0, which means full_seq_AMR is not given, so we won't skip.
                elif (cur_idx >= end_idx) and (end_idx > 0):
                    continue
                # if cur_idx is in the range of [start_idx, end_idx], but this residue has no CA, it is actually should be treat as missing residue.
                # We can preserve the information of this residue, but it should not take this residue into account when counting accepted residues.
                # elif has_ca_flag:
                #     accept_res_cnt += 1


                # het_flag, resseq, ic = res.id
                # if ic == ' ':
                #     ic = ''
                # if het_flag == ' ':
                #     het_flag = 'A'
                # if res.id[0] != ' ':
                #     # print('res.id[0] != ')
                #     # print(res.id[0])
                #     continue
                res_shortname = residue_constants.restype_3to1.get(res.resname, 'X'
                                                                )
                restype_idx = residue_constants.restype_order.get(res_shortname,
                                                                residue_constants.restype_num)
                atom_type_num = residue_constants.atom_type_num
                pos = np.zeros((atom_type_num, 3))
                atom2cgid = np.zeros((atom_type_num, 4)) # max cg num is 4 in a res 
                mask = np.zeros((atom_type_num,))
                res_b_factors = np.zeros((atom_type_num,))
                atom_name = np.empty((atom_type_num,), dtype=object)
                residue_name = np.empty((atom_type_num,), dtype=object)
                element = np.empty((atom_type_num,), dtype=object)
                ele2num = np.empty((atom_type_num,), dtype=object)
                hetfield = np.empty((atom_type_num,), dtype=object)
                resid = np.zeros((atom_type_num,))
                icode = np.empty((atom_type_num,), dtype=object)

                # print(cur_idx)
                for atom in res:
                    if atom.name not in residue_constants.atom_types:
                        continue
                    if ca_only:
                        if atom.name != 'CA':
                            continue

                    # get cg id from residue and atom
                    cg_id_type = [0,0,0,0]
                    # 暂时不需要用到cg
                    # cg_list = cg_constants.cg_dict[res.resname]
                    # for i in range(len(cg_list)):
                    #     if atom.name in cg_list[i]:
                    #         cg_id_type[i] = 1 

                    # residue_constants.atom_order[atom.name] is atom_id
                    atom2cgid[residue_constants.atom_order[atom.name]] = cg_id_type
                    pos[residue_constants.atom_order[atom.name]] = atom.coord
                    mask[residue_constants.atom_order[atom.name]] = 1
                    res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
                    atom_name[residue_constants.atom_order[atom.name]] = atom.name
                    residue_name[residue_constants.atom_order[atom.name]] = res.resname
                    element[residue_constants.atom_order[atom.name]] = atom.element
                    ele2num[residue_constants.atom_order[atom.name]] = residue_constants.ele2num[atom.element]
                    # hetfield[residue_constants.atom_order[atom.name]] = het_flag
                    resid[residue_constants.atom_order[atom.name]] = len(resids) + 1
                    # icode[residue_constants.atom_order[atom.name]] = ic

                if np.sum(mask) < 0.5:
                    continue
                aatype.append(restype_idx)
                cur_chain_aatype.append(restype_idx)
                atom_positions.append(pos)
                atom2cgids.append(atom2cgid)
                atom_mask.append(mask)
                residue_index.append(res.id[1])
                chain_ids.append(chain.id)
                b_factors.append(res_b_factors)
                atom_names.append(atom_name)
                residue_names.append(residue_name)
                elements.append(element)
                ele2nums.append(ele2num)
                # hetfields.append(hetfield)
                resids.append(resid)
                # icodes.append(icode)

            id2seq[chain.id] = cur_chain_aatype
        

        
        unique_chain_ids = np.unique(chain_ids)
        chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
        chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

        # if we only use CA atoms, then the accept_res_cnt should be the same as the number of nodes.
        if ca_only:
            assert accept_res_cnt == len(residue_names), "accept_res_cnt should be the same as the number of nodes. Check what's wrong."

        trans_target = []
        rot_m4_target = []
        rot_target = []
        alpha_C_trans_target = []
        alpha_C_rot_m4_target = []
        alpha_C_rot_target = []
        align_pos = []
        atom_trans = []
        rmsd = -1
        atom_model_trans = -1
        align_model_pos = -1
        # if data_type=='train':
            # here we would frist computer the R and T of each cg between start model and gt model
            # first obtein the superimpose, 
            
        re_gt_pos =torch.from_numpy(np.array(gt_atom_positions))
        # get the really res
        gt_atommaks = torch.from_numpy(np.array(gt_atom_mask))
        resmaks = torch.from_numpy(np.array(gt_res_mask))
        cor_af_pos = torch.from_numpy(np.array(atom_positions))
        cor_af_pos = cor_af_pos[resmaks==1]

        #find the backbone atom, where the index is 0,1,2,4
        res_list = torch.zeros(37)
        res_list[0], res_list[1], res_list[2], res_list[4] = 1,1,1,1

        # 获得初始化的backbone list
        gt_res_num = cor_af_pos.shape[0]
        backbone_list = res_list.repeat(gt_res_num, 1).view(-1)
        
        # 获取从atom mask中继续获得backbone原子的位置，其中1代表backbone原子
        finally_atom_mask = torch.logical_and(gt_atommaks.view(-1), backbone_list).int()

        # select the backbone atom mask to superimpose, the superimpose only change the backbone atom pos
        if cor_af_pos.shape[0] == re_gt_pos.shape[0]:
            rot_, tran_, rmsd, superimpose = superimpose_single(
                    reference= re_gt_pos.view(-1,3), 
                    coords= cor_af_pos.view(-1,3), 
                    mask=finally_atom_mask.view(-1) # 1 replace true that have this atom, else is miss.
                )
        else:
            print('gt shape:', re_gt_pos.shape)
            print('start shape:', torch.from_numpy(np.array(atom_positions)).shape)
            print('full_seq_AMR:',len(resmaks))
            print('we have res without missing:', resmaks.sum())
            return False


        reshape_superimpose = superimpose.view(-1,37,3)
        
        align_model_pos = reshape_superimpose
        atom_model_trans = (re_gt_pos-reshape_superimpose)

        if return_id2seq:
            return Protein(
                start2gt_model_tran=np.array(tran_), 
                start2gt_model_rot=np.array(rot_),
                trans_target=np.array(trans_target),
                atom_trans_align2gt=np.array(atom_model_trans),
                rot_target=np.array(rot_target),
                rot_m4_target=np.array(rot_m4_target),
                gt_res_mask=np.array(gt_res_mask),
                cdr1_mask=np.array(cdr1_mask),
                cdr2_mask=np.array(cdr2_mask),
                cdr3_mask=np.array(cdr3_mask),
                atom_positions=np.array(atom_positions),
                align_pos = np.array(align_model_pos),
                gt_atom_positions=np.array(gt_atom_positions),
                atom2cgids = np.array(atom2cgids),
                atom_mask=np.array(atom_mask),
                gt_atom_mask=np.array(gt_atom_mask),
                aatype=np.array(aatype),
                chain_index=chain_index,
                residue_index=np.array(residue_index),
                b_factors=np.array(b_factors),
                resolution=resolution
            ), id2seq
        else:
            return Protein(
                rmsd = np.array(rmsd),
                start2gt_model_tran=np.array(tran_), 
                start2gt_model_rot=np.array(rot_),
                trans_target=np.array(trans_target),
                atom_trans_align2gt=np.array(atom_model_trans),
                rot_target=np.array(rot_target),
                rot_m4_target=np.array(rot_m4_target),
                gt_res_mask=np.array(gt_res_mask),
                cdr1_mask=np.array(cdr1_mask),
                cdr2_mask=np.array(cdr2_mask),
                cdr3_mask=np.array(cdr3_mask),
                atom_positions=np.array(atom_positions),
                align_pos = np.array(align_model_pos),
                gt_atom_positions=np.array(gt_atom_positions),
                atom2cgids = np.array(atom2cgids),
                atom_mask=np.array(atom_mask),
                gt_atom_mask=np.array(gt_atom_mask),
                aatype=np.array(aatype),
                chain_index=chain_index,
                residue_index=np.array(residue_index),
                b_factors=np.array(b_factors),
                resolution=resolution,
                atom_names=np.array(atom_names),
                residue_names=np.array(residue_names),
                elements=np.array(elements),
                ele2nums=np.array(ele2nums),
                resids=np.array(resids).astype(int),
                chains=chains,
            )
            # reshape_atom2cgids = torch.from_numpy(np.array(atom2cgids))[resmaks==1]
            # for i in range(reshape_superimpose.shape[0]): # iter res
            #     res_all_pos_re = re_gt_pos[i] # gt model one_res_atom_pos
            #     # res_all_pos_re = cor_af_pos[i] # test the af model
            #     res_all_pos_reshape_superimpose = reshape_superimpose[i] # aligned from start model, one_res_atom_pos
            #     # save the aligned model pos
            #     align_pos.append(res_all_pos_reshape_superimpose.numpy())
                
            #     res_all_pos_reshape_atom2cgids = reshape_atom2cgids[i].transpose(0, 1) # [4, 37]
            #     res_all_pos_atommaks = atommaks[i] # [37]

            #     # save the trans between aligned model and gt model in each res pair
            #     res_atom_tran = res_all_pos_re - res_all_pos_reshape_superimpose
            #     atom_trans.append(res_atom_tran.numpy())

                # tran_tem = np.zeros((4, 3))
                # rot_tem = np.zeros((4, 9))
                # rot_m4_tem = np.zeros((4, 4))
                # alpha_C_tran_tem = np.zeros((4, 3))
                # alpha_C_rot_tem = np.zeros((4, 9))
                # alpha_C_rot_m4_tem = np.zeros((4, 4))

                # 分别获取superimpose res和gt res的(N,CA,C)构建的坐标系
                # gt_res_n_coord, gt_res_ca_coord, gt_res_c_coord = res_all_pos_re[0], res_all_pos_re[1], res_all_pos_re[2]
                # gt_res_rotation_matrix, gt_res_origin = get_local_coordinate_system(gt_res_n_coord, gt_res_ca_coord, gt_res_c_coord)
                
                # superimpose_res_n_coord, superimpose_res_ca_coord, superimpose_res_c_coord = res_all_pos_reshape_superimpose[0], res_all_pos_reshape_superimpose[1], res_all_pos_reshape_superimpose[2]
                # superimpose_res_rotation_matrix, superimpose_res_origin = get_local_coordinate_system(superimpose_res_n_coord, superimpose_res_ca_coord, superimpose_res_c_coord)
                
                #更新当前res的所有原子坐标换到局部坐标系上
                # local_frame_res_all_pos_re = transform_to_local_coordinates(res_all_pos_re, gt_res_origin, gt_res_rotation_matrix)
                # local_frame_res_all_pos_reshape_superimpose = transform_to_local_coordinates(res_all_pos_reshape_superimpose, superimpose_res_origin, superimpose_res_rotation_matrix)

                #测试是否能顺利从local 恢复到 global，检查可以恢复
                # restored_global_coords_res_all_pos_re = restore_to_global_coordinates(local_frame_res_all_pos_re, gt_res_origin, gt_res_rotation_matrix)
                # restored_global_coords_res_all_pos_reshape_superimpose = restore_to_global_coordinates(local_frame_res_all_pos_reshape_superimpose, superimpose_res_origin, superimpose_res_rotation_matrix)


                
                # for j in range(res_all_pos_reshape_atom2cgids.shape[0]): # iter cg
                #     cg_atom_index = res_all_pos_reshape_atom2cgids[j]
                #     if torch.sum(cg_atom_index) < 1: # if this cg is None
                #         continue
                #     #获得global坐标，与下面的local坐标，二选一，再计算r和t
                #     cg_atom_re_pos = res_all_pos_re[cg_atom_index==1] # get the cg atom in gt res
                #     cg_atom_superimpose_pos = res_all_pos_reshape_superimpose[cg_atom_index==1] # get the cg atom in gt res
                #     #获取local坐标（to do）


                #     cg_atommaks = res_all_pos_atommaks[cg_atom_index==1]
                #     # computer the r and t from cg_atom_superimpose_pos to cg_atom_re_pos
                #     rot_cg, tran_cg, rmsd_cg, superimposed_cg = superimpose_single(
                #         reference= cg_atom_re_pos, 
                #         coords=cg_atom_superimpose_pos, 
                #         mask=cg_atommaks
                #         )
                        
                #     tran_tem[j] = tran_cg.view(-1)
                #     rot_tem[j] = rot_cg.view(-1)

                #     r3 = R.from_matrix(rot_cg.squeeze(0))
                #     rot_m4_tem[j] = r3.as_quat()


                    # rot_si_cg, tran_si_cg, rmsd_si_cg, superimpose_si_cg = si(reference=cg_atom_re_pos, coords=cg_atom_superimpose_pos, mask=cg_atommaks)
                    # superimposed_cg_test = torch.einsum("bij, bmj -> bmi", rot_cg.unsqueeze(0), cg_atom_re_pos.unsqueeze(0)) + tran_cg.unsqueeze(0)

                    
                    # check again
                    # m4 = R.from_quat(rot_m4_tem[j]).as_matrix()

                # trans_target.append(tran_tem)
                # rot_target.append(rot_tem)
                # rot_m4_target.append(rot_m4_tem)
                # alpha_C_trans_target.append(alpha_C_tran_tem)
                # alpha_C_rot_target.append(alpha_C_rot_tem)
                # alpha_C_rot_m4_target.append(alpha_C_rot_m4_tem)

            
        # 计算真实的r和t
        # r = torch.from_numpy(np.array(rot_target))
        # t = torch.from_numpy(np.array(trans_target))
        # m4 = torch.from_numpy(np.array(rot_m4_target))

        # #计算atom2cg的矩阵
        # atom2cgids_tensor = torch.from_numpy(np.array(atom2cgids))[resmaks==1]
        # each_res_atom_times = torch.sum(atom2cgids_tensor, dim=-1)
        # # 转为形状 [batch_size, residue，atom=37, 1] 以进行广播
        # t_last = each_res_atom_times.unsqueeze(-1)
        # # 防止除以 0，将 t_last 中的 0 替换为一个很小的值（例如1e-6），避免计算错误
        # t_last_safe = torch.where(t_last == 0, torch.tensor(1e-6), t_last)
        
        
        # 先对r和t取平均再进行旋转平移。最后一个维度除以 t_last_safe 的最后一个值
        # mean_cg_trans_atom = (atom2cgids_tensor / t_last_safe)
        # atom_rot = torch.matmul(mean_cg_trans_atom, r).view(mean_cg_trans_atom.shape[0],mean_cg_trans_atom.shape[1],3,3)
        # atom_tran = torch.matmul(mean_cg_trans_atom, t)

        #打印第一个氨基酸的所有原子坐标，以及旋转平移矩阵
        # print(cor_af_pos[0])
        # print(reshape_superimpose[0])
        # print(re_gt_pos[0])
        # print(atom2cgids_tensor[0])
        # print(r[0])
        # print(t[0])
        # print(atom_rot[0])
        # print(atom_tran[0])
        
        #验证cg pair的r和t全部加到start model上，然后再与gt model比较
        # refine_init_model_bygt = torch.einsum("brij, brj -> bri", atom_rot, cor_af_pos)  + atom_tran
        #将更新后的坐标，再更新到start pdb上
        # result  = update_pdb_with_new_coords(structure, refine_init_model_bygt, resmaks, atommaks, "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/refine_data/refine_start_pdb.pdb")

        # rot_cor_r_t, tran_cor_r_t, rmsd_cor_r_t, superimpose_cor_r_t = superimpose_single(
        #         reference= re_gt_pos.view(-1,3), 
        #         coords= refine_init_model_bygt.view(-1,3), 
        #         mask=atommaks.view(-1) # 1 replace true that have this atom, else is miss.
        #     )
        
        #验证cg pair的r和t全部加到superimpose model上，然后再与gt model比较
        # refine_superimpose_model_bygt = torch.einsum("brij, brj -> bri", atom_rot, reshape_superimpose)  + atom_tran
        # result  = update_pdb_with_new_coords(structure, refine_superimpose_model_bygt, resmaks, atommaks, "/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/refine_data/refine_superimpose_pdb.pdb")

        # rot__imp_r_t, tran__imp_r_t, rmsd__imp_r_t, superimpose__imp_r_t = superimpose_single(
        #         reference= re_gt_pos.view(-1,3), 
        #         coords= refine_superimpose_model_bygt.view(-1,3), 
        #         mask=atommaks.view(-1) # 1 replace true that have this atom, else is miss.
        #     )

        # 分析扰动start model后，再加上r和t之后，与gt model之间的rmsd
        # noise_i = []
        # noise_rmsd = []
        # for i in range(20):
        #     # 生成高斯噪声，影响旋转角度，噪声标准差
        #     noise_std = 0.1 * (i+1)
        #     gaussian_noise = np.random.normal(loc=0.0, scale=noise_std, size=3)  # 对三个轴分别产生噪声

        #     # 原始的旋转角度 (绕x、y、z轴的旋转角度)
        #     base_rotation_angles = np.array([0.0, 0.0, 0.0])  # 初始没有旋转

        #     # 将噪声加入到旋转角度
        #     rotation_angles = base_rotation_angles + gaussian_noise

        #     # 生成旋转矩阵 (使用欧拉角表示旋转)
        #     rotation_matrix = torch.from_numpy(R.from_euler('xyz', rotation_angles).as_matrix())

        #     # 应用旋转矩阵到几何体的坐标上
        #     # rotated_geometry = geometry @ rotation_matrix.T
            
        #     noise_i.append(0.1 * (i+1))
        #     tran_update = tran_ * (0.1 * (i+1))
    
        #     # 加上噪声R和T
        #     superimposed_rot = torch.einsum("bij, bmj -> bmi", rotation_matrix.unsqueeze(0), cor_af_pos.view(1,-1,3))
        #     superimposed_R_T = superimposed_rot + tran_update.unsqueeze(0)

        #     # 加上真实的r和t
        #     superimposed_rot_r_t = torch.einsum('brij, brj -> bri', atom_rot, superimposed_R_T.view(-1,37,3))
        #     superimposed_r_t = superimposed_rot_r_t + atom_tran
        #     # superimposed_rot_r_t = torch.einsum("brij, brj -> bri", atom_rot, superimposed_R_T)
        #     # superimposed_r_t = superimposed_rot_r_t + atom_tran.unsqueeze(0)

        #     rot_noise, tran_noise, rmsd_noise, superimpose_noise = superimpose_single(
        #         reference= re_gt_pos.view(-1,3), 
        #         coords= superimposed_r_t.view(-1,3), 
        #         mask=atommaks.view(-1) # 1 replace true that have this atom, else is miss.
        #     )
        #     noise_rmsd.append(rmsd_noise)

        
        # sorted_lists = sorted(zip(noise_i, noise_rmsd))
        
        # sorted_list1, sorted_list2 = zip(*sorted_lists)
        # plt.plot(sorted_list1, sorted_list2, label='rmsd with noise', marker='o')

        # plt.title("RMSD after adding noise")
        # plt.xlabel("Noise")
        # plt.ylabel("RMSD")

        # plt.legend()

        # plt.savefig('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/rmsd_noise_sorted_plot.png')

        # gt_m4_rot = torch.from_numpy(np.array(rot_m4_target))
        # gt_rot = torch.from_numpy(np.array(rot_target))
        # gt_tran = torch.from_numpy(np.array(trans_target))
        # init_model = torch.from_numpy(np.array(atom_positions))
        # gt_model = torch.from_numpy(np.array(gt_atom_positions))
        # gt_res_mask = torch.from_numpy(np.array(gt_res_mask))
        # gt_atom_mask = torch.from_numpy(np.array(rot_target))
        # atom2cgids = torch.from_numpy(np.array(rot_target))


        


def is_antibody(seq, scheme='imgt', ncpu=4):
    """Run is_antibody method."""
    # code.
    """
    is_antibody:
    Args:
        seq : seq
        scheme : scheme
        ncpu : ncpu
    Returns:
    """
    seqs = [('0', seq)]
    numbering, alignment_details, hit_tables = anarci(seqs, scheme=scheme,
        output=False, ncpu=ncpu)
    if numbering[0] is None:
        return False, None
    if numbering[0] is not None and len(numbering[0]) > 1:
        logger.warning('There are %d domains in %s' % (len(numbering[0]), seq))
    chain_type = alignment_details[0][0]['chain_type'].lower()
    if chain_type is None:
        return False, None
    else:
        return True, chain_type
