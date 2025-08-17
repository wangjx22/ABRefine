"""Code."""
"""Protein data type."""

import sys
sys.path.append('/nfs_beijing_ai/jinxian/rama-scoring1.3.0')
import argparse
import os
os.environ["WANDB_IGNORE_GIT"] = "1"
import pandas as pd
import dataclasses
import io
import re
import collections
from typing import Any, Mapping, Optional, Sequence
import numpy as np
from np import residue_constants
from Bio.PDB import PDBParser
from anarci import anarci

import torch
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]
PICO_TO_ANGSTROM = 0.01
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)
from Bio.PDB import PDBParser, PDBIO, Atom
import time
import os
import os, sys
import argparse
import traceback
import sys
from pathlib import Path
from multiprocessing import Pool
import time
import io
from tqdm import tqdm
import tempfile
base_name = os.path.dirname(os.path.dirname(__file__))
sys.path.append(base_name)

import random
import numpy as np
import torch


import csv

def read_tsv_columns(file_path, column_names):
    """
    从TSV文件中读取指定列的值到各自的列表中
    
    参数:
        file_path (str): TSV文件路径
        column_names (list): 要读取的列名列表
    
    返回:
        dict: 列名作为键，对应值列表作为值的字典
    """
    # 初始化结果字典，每个列名对应一个空列表
    result = {col: [] for col in column_names}
    
    with open(file_path, 'r', encoding='utf-8') as file:
        # 创建TSV阅读器
        tsv_reader = csv.reader(file, delimiter='\t')
        
        # 读取标题行
        headers = next(tsv_reader)
        
        # 获取所需列的索引
        col_indices = {col: headers.index(col) for col in column_names if col in headers}
        
        # 检查是否所有请求的列都存在
        missing_cols = [col for col in column_names if col not in headers]
        if missing_cols:
            print(f"警告: 以下列不存在于文件中: {', '.join(missing_cols)}")
        
        # 逐行读取数据
        for row in tsv_reader:
            for col, idx in col_indices.items():
                if idx < len(row):  # 确保索引不越界
                    result[col].append(row[idx])
                else:
                    result[col].append('')  # 或者可以改为None
    
    return result



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
    # print('models:', models)
    model = models[0]
    return model, resolution, structure

def safe_float_convert(value, default=float('inf'), verbose=False):
    """
    增强版安全转换函数，处理多种非常规格式：
    - 包含逗号的分隔值（取第一个）
    - 多余的空格
    - 非数字字符串
    """
    try:
        # 预处理：移除首尾空格，替换中文逗号
        cleaned = str(value).strip().replace('，', ',')
        
        # 处理带逗号的情况（如 '3.9, 3.9'）
        if ',' in cleaned:
            first_part = cleaned.split(',')[0].strip()
            if verbose:
                print(f"检测到分隔值 '{cleaned}'，取第一个部分 '{first_part}'")
            return float(first_part)
            
        return float(cleaned)
        
    except ValueError:
        if verbose:
            print(f"无法转换的值: {value}，使用默认值 {default}")
        return default

def find_missing_chains(pdb_file):
    """Identify chains with missing residues based on REMARK 465 entries in a PDB file."""
    missing_chains = set()
    
    with open(pdb_file, 'r') as f:
        for line in f:
            # 仅处理REMARK 465行
            if not line.startswith("REMARK 465"):
                continue
                
            parts = line.strip().split()
            
            # 跳过不包含残基信息的行（至少需要3个数据字段：残基名+链ID+序号）
            if len(parts) < 5:  # "REMARK" + "465" + 至少3个数据字段
                continue
                
            data_fields = parts[2:]  # 移除"REMARK"和"465"
            
            # 按三个字段一组遍历（残基名, 链ID, 残基序号）
            for i in range(0, len(data_fields), 3):
                # 确保剩余字段足够组成一个完整的残基条目
                if i + 2 >= len(data_fields):
                    break
                
                chain_id = data_fields[i + 1]
                missing_chains.add(chain_id)
    
    return sorted(missing_chains)

# 使用示例
# if __name__ == "__main__":
#     pdb_path = "your_antibody.pdb"  # 替换为实际文件路径
#     chains_with_missing = find_missing_chains(pdb_path)
    
#     if chains_with_missing:
#         print(f"Chains with missing residues: {', '.join(chains_with_missing)}")
#     else:
#         print("No chains with missing residues found.")
import re
from collections import OrderedDict

three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'SEC': 'U', 'PYL': 'O', 'UNK': 'X'
}

def parse_resnum(res_num):
    """解析带插入码的残基编号 (如 100A)"""
    match = re.match(r"^(\d+)(\D*)", res_num.strip())
    if match:
        return (int(match.group(1)), match.group(2))
    return (0, "")

def check_missing_chains(pdb_file, chains):
    """检查指定链是否有缺失残基"""
    missing = set()
    with open(pdb_file) as f:
        for line in f:
            if line.startswith("REMARK 465"):
                parts = line[11:].split()
                for i in range(0, len(parts), 3):
                    if i+1 < len(parts) and parts[i+1] in chains:
                        missing.add(parts[i+1])
    return missing

def extract_chain_sequence(pdb_file, chain_id):
    """提取单个链的氨基酸序列"""
    residues = OrderedDict()
    
    with open(pdb_file) as f:
        current_model = 1
        for line in f:
            if line.startswith("MODEL"):
                current_model = int(line[10:14])
                continue
            if current_model > 1:
                continue
            
            if line.startswith("ATOM"):
                this_chain = line[21].strip()
                if this_chain != chain_id:
                    continue
                
                res_name = line[17:20].strip()
                res_num = line[22:27].strip()  # 包含插入码
                atom = line[12:16].strip()
                
                if atom == "CA" and res_name in three_to_one:
                    residues[res_num] = three_to_one[res_name]
    
    # 按残基编号排序
    sorted_res = sorted(residues.items(), key=lambda x: parse_resnum(x[0]))
    return "".join([aa for _, aa in sorted_res])

def process_antibody(pdb_path, Hchain, Lchain, output_path):
    """主处理函数"""
    # 参数校验
    if not Hchain and not Lchain:
        raise ValueError("At least one chain (H or L) must be specified")
    
    target_chains = []
    if Hchain: target_chains.append(Hchain)
    if Lchain: target_chains.append(Lchain)
    
    # 检查缺失残基
    missing = check_missing_chains(pdb_path, target_chains)
    if missing:
        print(f"Skipping antibody due to missing residues in chains: {', '.join(missing)}")
        return
    
    # 提取序列
    sequences = {}
    if Hchain:
        h_seq = extract_chain_sequence(pdb_path, Hchain)
        if not h_seq:
            print(f"No valid H-chain ({Hchain}) sequence found")
            return
        sequences['H'] = h_seq
    
    if Lchain:
        l_seq = extract_chain_sequence(pdb_path, Lchain)
        if not l_seq:
            print(f"No valid L-chain ({Lchain}) sequence found")
            return
        sequences['L'] = l_seq
    
    # 写入FASTA
    with open(output_path, 'w') as f:
        for chain_type, seq in sequences.items():
            chain_id = Hchain if chain_type == 'H' else Lchain
            header = f">antibody|pdb:{pdb_path[:-4]}|chain:{chain_id}|type:{chain_type}"
            f.write(f"{header}\n{seq}\n")
    
    print(f"Successfully saved antibody sequence to {output_path}")

# 使用示例
# if __name__ == "__main__":
#     process_antibody(
#         pdb_path="3td2.pdb",
#         Hchain="A",   # 直接指定重链标识符
#         Lchain="B",   # 直接指定轻链标识符
#         output_path="antibody_A_B.fasta"
#     )


def load_tsv(tsv_file):# 将sabdab AB的数据存为gen需要的格式数据，先分开生成，再给组装到一起
    title=['id','pdb','Hchain', 'Lchain', 'pdb_fpath', 'resolution', 'fasta_path']
    AB_No_miss_res_fasta_path = '/nfs_beijing_ai/jinxian/AB_No_miss_res_fasta_122524'
    os.system(f"mkdir -p {AB_No_miss_res_fasta_path}")
    #8be2_N,
    # 8be2,
    # N,
    # /pfs_beijing/ai_dataset/nbfold_dataset/pdb/be/8be2.pdb,
    # 1.9,
    # QVQLVESGGGLVQPGGSLRLSCAASRSISSINIMGWYRQAPGKERESVASHTRDGSTDYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCTTLTGFPRIRSWGQGTQVTVSSHHHHHHEPEA,
    # "['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127']",
    # "['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1', '1']",
    # vhh,
    # QVQLVESGGGLVQPGGSLRLSCAASRSISSINIMGWYRQAPGKERESVASHTRDGSTDYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCTTLTGFPRIRSWGQGTQVTVS[S],
    # 0.7948717948717948,
    # /nfs_beijing_ai/wanliang/playground/infer0415/output/nb15/8be2_N/ranked_unrelax_0.pdb,
    # /nfs_beijing_ai/wanliang/playground/infer0415/output/nb22/8be2_N/ranked_unrelax.pdb,
    # /nfs_beijing_ai/wanliang/playground/infer0415/output/vh15/8be2_N/ranked_unrelax_0.pdb,
    # /nfs_beijing_ai/wanliang/playground/infer0415/output/vh9/8be2_N/ranked_unrelax.pdb,
    # /nfs_beijing_ai/wanliang/playground/infer0415/output/vl11/8be2_N/ranked_unrelax.pdb,
    # 0

    # 读取 TSV 文件
    df = pd.read_csv(tsv_file, sep="\t")

    # 按列名提取数据
    column_names = df[["pdb", "Hchain", "Lchain", "resolution"]]
    sabdab_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SAbDab_AB/all_structures/imgt'

    
    file_path = tsv_file
    columns_to_read = ["pdb", "Hchain", "Lchain", "resolution"]  # 要读取的列名
    
    # 读取数据
    data = read_tsv_columns(file_path, columns_to_read)
    
    # 访问各个列的数据
    pdb = data['pdb']
    Hchain = data['Hchain']
    Lchain = data['Lchain']
    resolution = data['resolution']
    
    print(pdb[0])
    print(Lchain[0])
    print(resolution[0])
    length_data = len(pdb)
    print('length:', length_data)

    rec_list = []
    id_pdb_flag = [] # 不需要重复加入已有的pdb id   
    # for inx, enm in enumerate(resolution):
    for inx, enm in tqdm(enumerate(resolution), total=len(resolution)):
        # 安全转换分辨率
        res_value = safe_float_convert(
            resolution[inx], 
            default=999, 
            verbose=True
        )
        if Lchain[inx] == 'NA' or Hchain[inx] == "NA" or resolution[inx] == 'NOT' or res_value>4 :
            continue
        else:
            # 分别记录L和H链
            # 先判断pdb是否存在
            fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SAbDab_AB/all_structures/imgt/'+pdb[inx]+'.pdb'
            

            if os.path.exists(fpath):
                chains_with_missing = find_missing_chains(fpath)
    
                if chains_with_missing: #找到有确实氨基酸的链
                    print(f"Chains with missing residues: {', '.join(chains_with_missing)}")
                    if Lchain[inx] in chains_with_missing or Hchain[inx] in chains_with_missing:
                        #如果正好AB抗体的两条链中，存在有缺失氨基酸，则跳过
                        continue 
                else:
                    print("No chains with missing residues found.")
                    
                with open(fpath, "r") as f:
                    pdb_str = f.read()
                try:
                    
                    # model, resolution__, structure = get_model_from_str(pdb_str)
                    # full_seq = get_pdb_seq_by_CA(model, chain_id= Lchain[inx])

                    id_pdb = pdb[inx]+"_"+Hchain[inx]+"_"+Lchain[inx]
                    AB_fasta_path = AB_No_miss_res_fasta_path+'/' +id_pdb+'.fasta'

                    process_antibody(
                        pdb_path=fpath,
                        Hchain=Hchain[inx],   # 直接指定重链标识符
                        Lchain=Lchain[inx],   # 直接指定轻链标识符
                        output_path=AB_fasta_path
                    )

                    rec_list.append([id_pdb, pdb[inx], Hchain[inx], Lchain[inx], fpath, resolution[inx], AB_fasta_path])
                    id_pdb_flag.append(id_pdb)
                except Exception as e:
                    # 异常处理
                    print(f"\nError processing index {inx}: {str(e)}")
                    print('fpath:', fpath)
                    continue  # 继续下一个循环
                    
            else:
                continue
    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(rec_list, columns=title)
    df.to_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SAbDab_AB/sabdab_no_miss_res_AB.csv', index=False)
    
    # # 打印前5行作为示例
    # print("Names:", names[:5])
    # print("Ages:", ages[:5])
    # print("Emails:", emails[:5])

if __name__ == "__main__":
    load_tsv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SAbDab_AB/sabdab_summary_all.tsv')