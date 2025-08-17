import sys
sys.path.append('/nfs_beijing_ai/jinxian/rama-scoring1.3.0')
import argparse
import os
import random
from scipy.spatial.transform import Rotation as R

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import torch.optim as optim

from np.protein import from_pdb_string

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
from concurrent.futures import ProcessPoolExecutor

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
loss_mse = torch.nn.MSELoss(reduction='sum')
import wandb
import logging
from datetime import datetime


os.environ["WANDB_MODE"] = "run"


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

def find_imporvement_pdb_(csv_fpath='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522.csv', step_num=150637):
    df = load_data(csv_fpath)
    dic_pdb = {}
    # first sort by mean rmsd of each gt pdb
    for index, row in df.iterrows():
        pdb_name = row['pdb']  
        RMSD_CA = row['RMSD_CA']
        if pdb_name in dic_pdb:
            dic_pdb[pdb_name].append(RMSD_CA)
            
        else:
            dic_pdb[pdb_name] =  [RMSD_CA]
    # 计算每个键的平均值并重新赋值
    averages = {key: np.mean(values) for key, values in dic_pdb.items()}

    # 按照平均值从大到小排序
    sorted_averages = dict(sorted(averages.items(), key=lambda item: item[1], reverse=True))
    print(sorted_averages)

    # then count the imporvment pred pdb for sort list
    #'refine_CA_rmsd', 'start_rmsd_backbone', 'refine_rmsd_backbone', 'improve_num', 'improve_rat', 'reduced_num', 'reduced_rat'
    count_dict = {key: [0, 0, 0, 0, 0, 0, 0, 0, 0] for key in sorted_averages.keys()}
    
    # 文件路径
    fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/rmsd_values_2024.10.31-10.19.03_refine'
    file_path_1 = fpath + '_0.txt'
    file_path_2 = fpath + '_1.txt'

    # 读取两个文件到 DataFrame
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    # 合并 DataFrame
    df = pd.concat([df1, df2], ignore_index=True)
    # df = pd.read_csv(file_path)

    # 获得特定step的
    target_step = 'Step:'+str(step_num) # 替换为你想查找的 Step 值
    # 过滤出第一列中指定名字的所有行
    filtered_rows = df[df['Step'] == target_step]
    # 循环遍历每一行
    for index, row in filtered_rows.iterrows():
        # 根据列名获取指定值Step, start pdb name, start rmsd, refine rmsd, improve rat 
        start_pdb_name = row[' start pdb name']
        name =  start_pdb_name.strip("[]").split('/')[-2] 
        refine_CA_rmsd = row[' rmsd_refine_CA']
        start_rmsd = row[' start rmsd']
        refine_rmsd = row[' refine rmsd']
        # improve_rat = row[' improve rat ']
        
        # 'refine_CA_rmsd', 'refine_CA_improve', 'start_rmsd_backbone', 'refine_rmsd_backbone', 'improve_num', 'improve_rat', 'reduced_num', 'reduced_rat','refine_rmsd_backbone_imporve'
        if name in count_dict:
            count_dict[name][0] += refine_CA_rmsd
            count_dict[name][2] += start_rmsd
            count_dict[name][3] += refine_rmsd
            
            if start_rmsd > refine_rmsd:
                #有提升
                count_dict[name][4] += 1
                count_dict[name][5] += (start_rmsd - refine_rmsd)/start_rmsd
            else:
                count_dict[name][6] += 1
                count_dict[name][7] += (start_rmsd - refine_rmsd)/start_rmsd

def sort_pdb_by_mean_rmsd(csv_fpath='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522.csv', step_num=150637):
    df = load_data(csv_fpath)
    dic_pdb = {}
    # first sort by mean rmsd of each gt pdb
    for index, row in df.iterrows():
        pdb_name = row['pdb']  
        RMSD_CA = row['RMSD_CA']
        if pdb_name in dic_pdb:
            dic_pdb[pdb_name].append(RMSD_CA)
            
        else:
            dic_pdb[pdb_name] =  [RMSD_CA]
    # 计算每个键的平均值并重新赋值
    averages = {key: np.mean(values) for key, values in dic_pdb.items()}

    # 按照平均值从大到小排序
    sorted_averages = dict(sorted(averages.items(), key=lambda item: item[1], reverse=True))
    print(sorted_averages)

    # then count the imporvment pred pdb for sort list
    #'refine_CA_rmsd', 'start_rmsd_backbone', 'refine_rmsd_backbone', 'improve_num', 'improve_rat', 'reduced_num', 'reduced_rat'
    count_dict = {key: [0, 0, 0, 0, 0, 0, 0, 0, 0] for key in sorted_averages.keys()}
    
    # 文件路径
    fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/rmsd_values_2024.10.31-10.19.03_refine'
    file_path_1 = fpath + '_0.txt'
    file_path_2 = fpath + '_1.txt'

    # 读取两个文件到 DataFrame
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)

    # 合并 DataFrame
    df = pd.concat([df1, df2], ignore_index=True)
    # df = pd.read_csv(file_path)

    # 获得特定step的
    target_step = 'Step:'+str(step_num) # 替换为你想查找的 Step 值
    # 过滤出第一列中指定名字的所有行
    filtered_rows = df[df['Step'] == target_step]
    # 循环遍历每一行
    for index, row in filtered_rows.iterrows():
        # 根据列名获取指定值Step, start pdb name, start rmsd, refine rmsd, improve rat 
        start_pdb_name = row[' start pdb name']
        name =  start_pdb_name.strip("[]").split('/')[-2] 
        refine_CA_rmsd = row[' rmsd_refine_CA']
        start_rmsd = row[' start rmsd']
        refine_rmsd = row[' refine rmsd']
        # improve_rat = row[' improve rat ']
        
        # 'refine_CA_rmsd', 'refine_CA_improve', 'start_rmsd_backbone', 'refine_rmsd_backbone', 'improve_num', 'improve_rat', 'reduced_num', 'reduced_rat','refine_rmsd_backbone_imporve'
        if name in count_dict:
            count_dict[name][0] += refine_CA_rmsd
            count_dict[name][2] += start_rmsd
            count_dict[name][3] += refine_rmsd
            
            if start_rmsd > refine_rmsd:
                #有提升
                count_dict[name][4] += 1
                count_dict[name][5] += (start_rmsd - refine_rmsd)/start_rmsd
            else:
                count_dict[name][6] += 1
                count_dict[name][7] += (start_rmsd - refine_rmsd)/start_rmsd
            

    print(count_dict)
    # 遍历字典中的每个键，并更新统计数值
    for key, value_list in count_dict.items():
        value_list[0] = value_list[0] / (value_list[4]+value_list[6]+1e-8) # mean refine CA rmsd 
        value_list[1] = ( sorted_averages[key]-value_list[0] )/sorted_averages[key]
        value_list[2] = value_list[2] / (value_list[4]+value_list[6]+1e-8) # mean start_rmsd  
        value_list[3] = value_list[3] / (value_list[4]+value_list[6]+1e-8) # mean refine_rmsd
        value_list[8] = (value_list[2]-value_list[3])/(value_list[2]+1e-8)
        
        if value_list[4] !=0:
            value_list[5] = value_list[5] / value_list[4] # 更新提升值
        
        if value_list[6] !=0:
            value_list[7] = value_list[7] / value_list[6] # 更新降低值

    print(count_dict)

    # 合并字典
    rows = []
    for key, value_list in sorted_averages.items():
        rows.append([key, value_list, count_dict[key][0], count_dict[key][1],count_dict[key][2],count_dict[key][3],count_dict[key][8],count_dict[key][4],count_dict[key][5],count_dict[key][6],count_dict[key][7]])

    column_names = ['pdb_name', 'start_CA_rmsd_from_file', 'refine_CA_rmsd', 'refine_CA_improve_rat', 'start_backbone_rmsd', 'refine_backbone_rmsd', 'refine_backbone_rmsd_imporve', 'improve_num', 'improve_rat', 'reduced_num', 'reduced_rat']

    # 将列表转换为 DataFrame
    df = pd.DataFrame(rows, columns=column_names)

    # 存储到 CSV 文件
    output_file = fpath + '_data_analysis_step_'+str(step_num)+'.csv'
    df.to_csv(output_file, index=False)


def inference_sort_pdb_by_mean_rmsd():
    
    # 文件路径
    fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/inference_rmsd_values_run_train_sin_k8s_trial3_2024.11.18-12.13.00_Step_100000_improve_rat_0.0831341980535005_imporve_num_168_model_zoo_nb_no4_95_testset_0522_new.txt'

    df1 = pd.read_csv(fpath)
    imporve_list = []
    my_ca_rmsd = []
    my_ca_rmsd_imp = []
    my_ca_rmsd_refine =[]
    for index, row in df1.iterrows():
        
        start_pdb_name = row['start pdb name']
        name =  start_pdb_name.strip("[]").split('/')[-2] 
        rmsd_start_CA = row[' rmsd_start_CA']
        my_ca_rmsd.append(rmsd_start_CA)
        rmsd_CA_improve_rat = row[' rmsd_CA_improve_rat']
        my_ca_rmsd_imp.append(rmsd_CA_improve_rat)
        rmsd_refine_CA = row[' rmsd_refine_CA']
        my_ca_rmsd_refine.append(rmsd_refine_CA)
        

    refine_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test_data_results/nb-rf-subset-tmp-metric-report_all.csv'
    refine_df = load_data(refine_path)
    refine_df_RMSD_CA = list(refine_df["RMSD_CA"])
    for i in range(len(refine_df_RMSD_CA)):
        if refine_df_RMSD_CA[i] < my_ca_rmsd[i]:
            imporve_list.append(float(my_ca_rmsd[i]-refine_df_RMSD_CA[i])/my_ca_rmsd[i])
            print(my_ca_rmsd[i], refine_df_RMSD_CA[i], (my_ca_rmsd[i]-refine_df_RMSD_CA[i])/my_ca_rmsd[i], my_ca_rmsd_refine[i], my_ca_rmsd_imp[i])

    print(len(imporve_list))
    print(sum(imporve_list)/len(imporve_list))

    # print(len(imporve_list))
    # float_values = [item[1] for item in imporve_list]
    # average_float = sum(float_values) / len(float_values)

    # unique_names = set(item[0] for item in imporve_list)
    # print(average_float)
    # print(unique_names)


    

    # 合并 DataFrame
    # df = pd.read_csv(file_path)

def save_fig(rmsd_CA_CDR_refine_improve_with_CDR_weight,rmsd_CA_CDR_refine_improve_without_CDR_weight, CDR_region):
    # 从上已经获得是否使用CDR weight的结果。下面开始分析，计算每个pdb的在使用cdr weight前后的rmsd变化
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

    # 横坐标序号列表，与指标list长度一致，用于替代pdb名称
    x_axis = range(len(rmsd_CA_CDR_refine_improve_with_CDR_weight))

    # 创建图形对象，设置图形的宽度和高度，这里将宽度增大（单位为英寸，可根据需要调整数值）
    fig = plt.figure(figsize=(20, 6))

    plot_with_CDR_weight_list = rmsd_CA_CDR_refine_improve_with_CDR_weight
    plot_without_CDR_weight_list = rmsd_CA_CDR_refine_improve_without_CDR_weight

    # 在图形对象上添加子图
    ax = fig.add_subplot(111)
    with_CDR_weight_mean = sum(plot_with_CDR_weight_list) / len(plot_with_CDR_weight_list)
    ax.plot(x_axis, plot_with_CDR_weight_list, marker='o', label=f'with_cdr and the mean rmsd is {with_CDR_weight_mean} ')
    without_CDR_weight_mean = sum(plot_without_CDR_weight_list) / len(plot_without_CDR_weight_list)
    ax.plot(x_axis, plot_without_CDR_weight_list, marker='o', label=f'without_cdr and the mean rmsd is {without_CDR_weight_mean}')

    plt.title('The '+ CDR_region +' rmsd_CA_CDR_refine_improve metric')
    plt.xlabel('pdb index')
    plt.ylabel('metric')

    # 添加图例，用于区分不同的折线
    plt.legend()

    # 自动调整子图参数，以适应图形元素，防止标签被截断等情况
    plt.tight_layout()

    # 保存图片，可根据需求修改保存的文件名和格式（这里保存为png格式）
    plt.savefig('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/process_data_rmsd_CA_CDR_refine_improve_region_'+CDR_region+'.png')

def is_CDR_loss_work(file_with_CDR_weight, file_without_CDR_weight):
    # 分析CDR loss weight是否起到了优化CDR的作用。
    # 先分别获得未使用CDR weight和使用的结果
    # file_without_CDR_weight = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test/refinefile_model_zoo_nb_no4_95_testset_0522_refinemodel_2024.10.31-10.19.03_Step_136637_improve_rat_0.0506243271965237_imporve_num_343_yaml_name_run_train_sin_k8s_trial1.txt'
    # file_with_CDR_weight = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test/refinefile_model_zoo_nb_no4_95_testset_0522_refinemodel_run_train_sin_k8s_trial3_2024.11.27-03.01.42_Step_322000_improve_rat_0.0753710768617964_imporve_num_260_yaml_name_run_train_sin_k8s_trial3.txt'
    
    df1 = pd.read_csv(file_without_CDR_weight)
    rmsd_start_CA_without_CDR_weight = []
    rmsd_refine_CA_without_CDR_weight = []
    rmsd_CA_improve_without_CDR_weight = []
    rmsd_refine_CA_CDR_before_without_CDR_weight = []
    rmsd_refine_CA_CDR_after_without_CDR_weight = []
    rmsd_CA_CDR_refine_improve_without_CDR_weight = []
    CDR1_impove_rmsd_without_CDR_weight = []
    CDR2_impove_rmsd_without_CDR_weight = []
    CDR3_impove_rmsd_without_CDR_weight = []

    for index, row in df1.iterrows():
        start_pdb_name = row['start_pdb_fpath']
        name =  start_pdb_name.strip("[]").split('/')[-2] 
        rmsd_start_CA_without_CDR_weight.append(row[' rmsd_start_CA'])
        rmsd_refine_CA_without_CDR_weight.append(row[' rmsd_refine_CA'])
        rmsd_CA_improve_without_CDR_weight.append(row[' rmsd_CA_improve'])
        rmsd_refine_CA_CDR_before_without_CDR_weight.append(row[' rmsd_refine_CA_CDR_before'])
        rmsd_refine_CA_CDR_after_without_CDR_weight.append(row[' rmsd_refine_CA_CDR_after'])
        rmsd_CA_CDR_refine_improve_without_CDR_weight.append(row[' rmsd_CA_CDR_refine_improve'])
        CDR1_impove_rmsd_without_CDR_weight.append(row[' CDR1_impove_rmsd'])
        CDR2_impove_rmsd_without_CDR_weight.append(row[' CDR2_impove_rmsd'])
        CDR3_impove_rmsd_without_CDR_weight.append(row[' CDR3_impove_rmsd'])

    df1 = pd.read_csv(file_with_CDR_weight)
    rmsd_start_CA_with_CDR_weight = []
    rmsd_refine_CA_with_CDR_weight = []
    rmsd_CA_improve_with_CDR_weight = []
    rmsd_refine_CA_CDR_before_with_CDR_weight = []
    rmsd_refine_CA_CDR_after_with_CDR_weight = []
    rmsd_CA_CDR_refine_improve_with_CDR_weight = []
    CDR1_impove_rmsd_with_CDR_weight = []
    CDR2_impove_rmsd_with_CDR_weight = []
    CDR3_impove_rmsd_with_CDR_weight = []

    for index, row in df1.iterrows():
        rmsd_start_CA_with_CDR_weight.append(row[' rmsd_start_CA'])
        rmsd_refine_CA_with_CDR_weight.append(row[' rmsd_refine_CA'])
        rmsd_CA_improve_with_CDR_weight.append(row[' rmsd_CA_improve'])
        rmsd_refine_CA_CDR_before_with_CDR_weight.append(row[' rmsd_refine_CA_CDR_before'])
        rmsd_refine_CA_CDR_after_with_CDR_weight.append(row[' rmsd_refine_CA_CDR_after'])
        rmsd_CA_CDR_refine_improve_with_CDR_weight.append(row[' rmsd_CA_CDR_refine_improve'])
        CDR1_impove_rmsd_with_CDR_weight.append(row[' CDR1_impove_rmsd'])
        CDR2_impove_rmsd_with_CDR_weight.append(row[' CDR2_impove_rmsd'])
        CDR3_impove_rmsd_with_CDR_weight.append(row[' CDR3_impove_rmsd'])

    save_fig(rmsd_CA_CDR_refine_improve_with_CDR_weight, rmsd_CA_CDR_refine_improve_without_CDR_weight, 'All_CDR')
    save_fig(CDR1_impove_rmsd_with_CDR_weight, CDR1_impove_rmsd_without_CDR_weight, 'CDR1')
    save_fig(CDR2_impove_rmsd_with_CDR_weight, CDR2_impove_rmsd_without_CDR_weight, 'CDR2')
    save_fig(CDR3_impove_rmsd_with_CDR_weight, CDR3_impove_rmsd_without_CDR_weight, 'CDR3')

import csv
from np.protein import from_pdb_string, get_model_from_str, get_pdb_seq_by_CA

def get_seq(gt_pdb_filepath):
    with open(gt_pdb_filepath, 'r') as f:
            gt_pdb_str = f.read()
    gt_model, _, _structure = get_model_from_str(gt_pdb_str)
    gt_pdb_seq = get_pdb_seq_by_CA(gt_model, chain_id=None)
    return gt_pdb_seq

def process_AFDB(AFDB_train_csv, target_file):
    # 初始化一个空的列表，用于存储第一列数据
    first_column_list = []

    # 打开 lst 文件并逐行读取
    with open(target_file, "r") as file:
        for line in file:
            # 去掉行尾换行符并分割每一行的列
            columns = line.strip().split()
            # 如果该行包含至少一列数据，提取第一列
            if columns:
                first_column_list.append(columns[0])


    # 读取CSV文件
    df = pd.read_csv(AFDB_train_csv)
    # 获取指定列，例如"column_name"列
    pdb_fpath = df["pdb_fpath"].tolist()  # 转为list格式
    pdb_fpath_gt = df["pdb_fpath_gt"].tolist()  # 转为list格式
    
    fin_target = []
    fin_num_flag = 0
    #从target_file找到对应的pdb
    for target_pdb in first_column_list:
        matching_indices = [i for i, s in enumerate(pdb_fpath) if target_pdb in s]
        if len(matching_indices) > 0:
            #继续判断pdb seq是否相等
            gt_seq = get_seq(pdb_fpath_gt[matching_indices[0]])
            seq = get_seq(pdb_fpath[matching_indices[0]])
            if gt_seq == seq:
                fin_target.append([pdb_fpath_gt[matching_indices[0]], pdb_fpath[matching_indices[0]]])
                fin_num_flag += 1
                print(fin_num_flag)
            else:
                print('The seq not equ:', target_pdb)
        else:
            print('Not find:', target_pdb)

    # 将数据写入 CSV 文件
    print(AFDB_train_csv.split('.')[0]+'_new.csv')
    with open(AFDB_train_csv.split('.')[0]+'_new.csv', mode="w", newline="") as file:
        writer = csv.writer(file)
        # 写入每行
        writer.writerows(fin_target)

from Bio.PDB import PDBParser
import numpy as np

def calculate_ca_rmsd(pdb_path_1, pdb_path_2):
    # 初始化PDB解析器
    parser = PDBParser(QUIET=True)
    
    # 解析两个PDB文件
    structure1 = parser.get_structure("structure1", pdb_path_1)
    structure2 = parser.get_structure("structure2", pdb_path_2)
    
    # 提取Cα坐标
    ca_atoms_1 = [atom for atom in structure1.get_atoms() if atom.get_id() == "CA"]
    ca_atoms_2 = [atom for atom in structure2.get_atoms() if atom.get_id() == "CA"]

    # 检查Cα原子数量是否一致
    if len(ca_atoms_1) != len(ca_atoms_2):
        raise ValueError("The number of Cα atoms in the two PDB files does not match.")
    
    # 计算RMSD
    differences = np.array([atom1.coord - atom2.coord for atom1, atom2 in zip(ca_atoms_1, ca_atoms_2)])
    rmsd = np.sqrt(np.mean(np.sum(differences**2, axis=1)))
    
    return rmsd

import matplotlib.pyplot as plt

def sis_CA_rmsd(AFDB_train_csv):
    # 读取CSV文件
    df = pd.read_csv(AFDB_train_csv)
    # 获取指定列，例如"column_name"列
    pdb_fpath = df["pdb_fpath"].tolist()  # 转为list格式
    pdb_fpath_gt = df["pdb_fpath_gt"].tolist()  # 转为list格式
    rmsd_values = []
    pass_rmsd0 = []
    ta = 1
    for i in range(len(pdb_fpath_gt)):
        rmsd_ = calculate_ca_rmsd(pdb_fpath[i], pdb_fpath_gt[i])
        rmsd_values.append(rmsd_)
        print(ta)
        ta += 1
        
        pass_rmsd0.append([pdb_fpath[i], pdb_fpath_gt[i], rmsd_])
    
    with open('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/AFDB_train_dataset_with_rmsd.csv', mode="w", newline="") as file:
        writer = csv.writer(file)
        # 写入每行
        writer.writerows(pass_rmsd0)
    
    # 绘制直方图

    # # 计算统计量
    # mean_rmsd = np.mean(rmsd_values)
    # median_rmsd = np.median(rmsd_values)
    # mode_rmsd = np.argmax(np.bincount(np.round(rmsd_values, 2).astype(int)))  # 重峰值

    # # 绘制直方图
    # plt.figure(figsize=(12, 7))
    # plt.hist(rmsd_values, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    # plt.xlabel("RMSD (Å)")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of RMSD values")
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # # 添加平均值、中位数、重峰值标记
    # plt.axvline(mean_rmsd, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_rmsd:.2f}')
    # plt.axvline(median_rmsd, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_rmsd:.2f}')
    # plt.axvline(mode_rmsd, color='purple', linestyle='dashed', linewidth=1.5, label=f'Mode: {mode_rmsd:.2f}')

    # # 添加图例
    # plt.legend()

    # # 保存图像
    # plt.savefig("/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/AFDB_train_dataset_CA_rmsd.png", dpi=300)
def valid_sabdab(pdb_file_path):
    with open(pdb_file_path, 'r') as f:
        pdb_str = f.read()
    model, resolution, structure = get_model_from_str(pdb_str)
    pdb_seq = get_pdb_seq_by_CA(model, chain_id='D')
    print(len(pdb_seq))

def split_data(train_data_fpath):
    df = load_data(model_zoom_fpath)
    model_zoom_pdb_id = df['pdb_id'] 
    #先统计有多少pdb_id,然后按照序列相似性划分train，validation，和test

import shutil
import csv
import os
def copy_files_from_csv(csv_path, target_folder):
    """
    从CSV文件中读取gt_file列的文件路径 并将文件复制到指定文件夹。

    参数:
    csv_path (str): CSV文件的路径。
    target_folder (str): 目标文件夹路径 用于存放复制过来的文件。
    """
    # csv_file_path = "your_csv_file.csv"  # 替换为实际的CSV文件路径
    # destination_folder = "destination_folder"  # 替换为实际的目标文件夹路径
    # 确保目标文件夹存在 不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_path = row['pdb_fpath_gt']
            if os.path.isfile(file_path):
                file_name = os.path.basename(file_path)
                target_path = os.path.join(target_folder, file_name)
                shutil.copy2(file_path, target_path)
            else:
                print(f"{file_path} 不是一个有效的文件路径，跳过复制。")

def statice_what_kind_model_gen_better_models(result_file):
    # 分析结果中，11个模型生成的decoys，哪种提升最大
    # 根据生成模型进行划分11个子集统计数据
    af_v1m1 = [] #-0.09996311129494147
    af_v1m2 = [] # -0.08623400669206273
    af_v1m3 = []#-0.07802204008806836
    af_v1m4 = []#-0.07277886366302316
    openfold_wl = [] # -0.0581641515547579
    _328_absolute_evo2_amr = [] #-0.1324854310263287
    _328_evo8_amr = [] #-0.12966461750594052
    _0320_absolute_numbering_resnet_with_dropout0p2_lr_5e_4 = [] #-0.10652787644754756
    _0401_scratch_rmsd_weighted = [] #-0.15564041991125452
    _0401_scratch_run3 = []# -0.0.13546576554125006
    # _3loss_ssfape0.1_wCdr_nb_distill_v2.3_fix = [] #-0.1388801329515197


    df1 = pd.read_csv(result_file)
    rmsd_CA_improve = []
    f = 0 
    for index, row in df1.iterrows():
        rmsd_CA_improve.append(row[' rmsd_CA_improve'])
        if (index+1) % 88 == 0 and f ==1 :
            print(row['start_pdb_fpath'])
            print(sum(af_v1m1)/88)
            
            af_v1m1=[]
        else:
            if f==0:
                f=1
            af_v1m1.append(row[' rmsd_CA_improve'])
    print('the mean rmsd ca:', sum(rmsd_CA_improve)/len(rmsd_CA_improve))
    rmsd_start_CA_without_CDR_weight = []
    rmsd_refine_CA_without_CDR_weight = []
    rmsd_CA_improve_without_CDR_weight = []
    rmsd_refine_CA_CDR_before_without_CDR_weight = []
    rmsd_refine_CA_CDR_after_without_CDR_weight = []
    rmsd_CA_CDR_refine_improve_without_CDR_weight = []
    CDR1_impove_rmsd_without_CDR_weight = []
    CDR2_impove_rmsd_without_CDR_weight = []
    CDR3_impove_rmsd_without_CDR_weight = []

    for index, row in df1.iterrows():
        start_pdb_name = row['start_pdb_fpath']
        name =  start_pdb_name.strip("[]").split('/')[-2] 
        rmsd_start_CA_without_CDR_weight.append(row[' rmsd_start_CA'])
        rmsd_refine_CA_without_CDR_weight.append(row[' rmsd_refine_CA'])
        rmsd_CA_improve_without_CDR_weight.append(row[' rmsd_CA_improve'])
        rmsd_refine_CA_CDR_before_without_CDR_weight.append(row[' rmsd_refine_CA_CDR_before'])
        rmsd_refine_CA_CDR_after_without_CDR_weight.append(row[' rmsd_refine_CA_CDR_after'])
        rmsd_CA_CDR_refine_improve_without_CDR_weight.append(row[' rmsd_CA_CDR_refine_improve'])
        CDR1_impove_rmsd_without_CDR_weight.append(row[' CDR1_impove_rmsd'])
        CDR2_impove_rmsd_without_CDR_weight.append(row[' CDR2_impove_rmsd'])
        CDR3_impove_rmsd_without_CDR_weight.append(row[' CDR3_impove_rmsd'])


def create_esmflow_NB_data(input_csv_path, output_csv_path):
    with open(input_csv_path, 'r', encoding='utf-8') as input_file, \
            open(output_csv_path, 'w', encoding='utf-8', newline='') as output_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            old_path = row['pdb_fpath']
            pdb_name = row['pdb']
            new_path = '/nfs_beijing_ai/qiuzy/git_repos/esmflow/v3.5/eval/pred_pdbs/'+pdb_name+'_0.pdb'
            row['pdb_fpath'] = new_path
            writer.writerow(row)

from np.protein import from_pdb_string
def prepare_features(gt_pdb_filepath, af_pdb_filepath, chain, full_seq_AMR, data_type='train', dataset_type='NB'):
    """Run prepare_features method.
    gt_pdb_filepath: a file path of gt_pdb_filepath
    af_pdb_filepath: a file path of af_pdb_filepath
    chain: the chain of pdb

    output: a torch_geometric data of the pdb
    """
    
    # code.
    
    with open(af_pdb_filepath, 'r') as f:
        pdb_str = f.read()
    
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

    
    #find the backbone atom, where the index is 0,1,2,4
    res_list = torch.zeros(37)
    res_list[0], res_list[1], res_list[2], res_list[4] = 1,1,1,1

    # align data
    align_atom_positions = torch.from_numpy(eval(f"po.{'align_pos'}").astype(np.float32))
    # true_align_atom_positions = align_atom_positions.view(-1,3)[gt_atom_mask.view(-1)==1]

    start_atom_mask = po.atom_mask
    gt_res_mask = torch.from_numpy(eval(f"po.{'gt_res_mask'}"))
    
    atom_mask = start_atom_mask[gt_res_mask==1].reshape(-1).astype(bool)
    atom_mask = torch.from_numpy(atom_mask)

    # 获得初始化的backbone list
    gt_res_num = align_atom_positions.shape[0]
    backbone_list = res_list.repeat(gt_res_num, 1).view(-1)
    
    # 获取从atom mask中继续获得backbone原子的位置，其中1代表backbone原子
    finally_atom_mask = torch.logical_and(atom_mask, backbone_list).int()
    
    atom_trans_align2gt = torch.from_numpy(eval(f"po.{'atom_trans_align2gt'}").astype(np.float32))
    align_atom_trans = atom_trans_align2gt.view(-1,3)[finally_atom_mask==1]
    cdr1_mask = torch.from_numpy(eval(f"po.{'cdr1_mask'}"))
    cdr2_mask = torch.from_numpy(eval(f"po.{'cdr2_mask'}"))
    cdr3_mask = torch.from_numpy(eval(f"po.{'cdr3_mask'}"))

    # frist select the mask from res_mask
    cdr1_mask_by_res_mask = cdr1_mask[gt_res_mask==1]
    cdr2_mask_by_res_mask = cdr2_mask[gt_res_mask==1]
    cdr3_mask_by_res_mask = cdr3_mask[gt_res_mask==1]
    cdr_mask_by_res_mask = cdr1_mask_by_res_mask+cdr2_mask_by_res_mask+cdr3_mask_by_res_mask
    atom_trans_align2gt_cdr_mask  = atom_trans_align2gt[cdr_mask_by_res_mask==1]
    
    cdr_res_num = sum(cdr_mask_by_res_mask)
    cdr_res_backbone_list = res_list.repeat(cdr_res_num, 1).view(-1)
    
    # 获取从atom mask中继续获得backbone原子的位置，其中1代表backbone原子
    cdr_atom_mask = start_atom_mask[gt_res_mask==1][cdr_mask_by_res_mask==1].reshape(-1).astype(bool)
    cdr_atom_mask = torch.from_numpy(cdr_atom_mask)
    finally_cdr_atom_mask = torch.logical_and(cdr_atom_mask, cdr_res_backbone_list).int()
    cdr_tran = atom_trans_align2gt_cdr_mask.view(-1,3)[finally_cdr_atom_mask==1]
    return align_atom_trans, cdr_tran 


import seaborn as sns
def analyze_distribution_pdb_trans(csv_file):
    #分析每个gt pdb和其decoys的tran的分布情况
    df = load_data(csv_file)
    
    pdb_id = list(df["id"])
    pdb_files = list(df["pdb_fpath"])
    gt_pdb_files = list(df["pdb_fpath_gt"])
    chain_ids = list(df["chain"])
    full_seq_amr = list(df["full_seq_AMR"])

    # 4x7d_C，先暂时分析一个
    data = []
    data_7xtp_C = []
    flag = 0
    flag_7xtp_C =0
    for idx in range(len(pdb_files)):
        gt_fpath = gt_pdb_files[idx]
        af_fpath = pdb_files[idx]
        chain_id = chain_ids[idx]
        full_seq_amr_ = full_seq_amr[idx]
        id = pdb_id[idx]
        if id == '4x7d_C' and flag < 60:
            trans_, cdr_tran = prepare_features(
                gt_pdb_filepath=gt_fpath, 
                af_pdb_filepath=af_fpath, 
                chain=chain_id, 
                full_seq_AMR=full_seq_amr_
                )
            data.append(cdr_tran)
            flag+=1
        
        if id =='7xtp_C' and flag_7xtp_C < 60:
            trans_, cdr_tran = prepare_features(
                gt_pdb_filepath=gt_fpath, 
                af_pdb_filepath=af_fpath, 
                chain=chain_id, 
                full_seq_AMR=full_seq_amr_
                )
            data_7xtp_C.append(cdr_tran)
            flag_7xtp_C+=1
        
        if flag_7xtp_C ==60 and flag == 60:
            break
    # 2. 整体数据合并
    #-----------------------------
    all_pointsA = np.concatenate(data, axis=0)  # shape: [NA, 3]
    all_pointsB = np.concatenate(data_7xtp_C, axis=0)  # shape: [NB, 3]
    print("数据 A 合并后 shape:", all_pointsA.shape)
    print("数据 B 合并后 shape:", all_pointsB.shape)
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制 A
    ax.scatter(all_pointsA[:,0], all_pointsA[:,1], all_pointsA[:,2],
            c='blue', alpha=0.6, label='Data A')

    # 绘制 B
    ax.scatter(all_pointsB[:,0], all_pointsB[:,1], all_pointsB[:,2],
            c='red', alpha=0.6, label='Data B')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot: Data A vs Data B')
    ax.legend()
    plt.savefig('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/4x7d_C_and_7xtp_C_distrbution.png')



    dfA = pd.DataFrame(all_pointsA, columns=['X','Y','Z'])
    dfA['Category'] = 'A'

    dfB = pd.DataFrame(all_pointsB, columns=['X','Y','Z'])
    dfB['Category'] = 'B'

    df_concat = pd.concat([dfA, dfB], axis=0, ignore_index=True)

    sns.pairplot(df_concat, hue='Category', diag_kind='kde', corner=True, 
                plot_kws={'alpha':0.6})
    plt.suptitle("Pair Plot: Data A vs Data B (2D投影对比)", y=1.02)
    plt.show()
    plt.savefig('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/4x7d_C_and_7xtp_C_distrbution_PairPlot.png')


    dims = ['X', 'Y', 'Z']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    for i in range(3):
        ax = axes[i]
        
        # 绘制数据 A 的直方图 & KDE
        sns.histplot(all_pointsA[:, i], kde=True, color='blue', alpha=0.4, 
                    label='Data A', ax=ax, stat='density')
        
        # 绘制数据 B 的直方图 & KDE
        sns.histplot(all_pointsB[:, i], kde=True, color='red', alpha=0.4,
                    label='Data B', ax=ax, stat='density')
        
        ax.set_title(f"Dimension: {dims[i]}")
        ax.legend()

    plt.tight_layout()
    plt.savefig('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/4x7d_C_and_7xtp_C_distrbution_KDE.png')


    meanA = np.mean(all_pointsA, axis=0)
    stdA  = np.std(all_pointsA, axis=0)

    meanB = np.mean(all_pointsB, axis=0)
    stdB  = np.std(all_pointsB, axis=0)

    print(f"Data A 均值: {meanA}, 标准差: {stdA}")
    print(f"Data B 均值: {meanB}, 标准差: {stdB}")




if __name__ == '__main__':
    # statice_what_kind_model_gen_better_models('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test/refinefile_esmfolw_NB_new_refinemodel_run_train_sin_k8s_trial3_2024.11.27-03.01.42_Step_322000_improve_rat_0.0753710768617964_imporve_num_260_yaml_name_run_train_sin_k8s_trial3.txt')
    # is_CDR_loss_work(file_with_CDR_weight='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test/refinefile_model_zoo_nb_no4_95_testset_0522_refinemodel_run_train_sin_k8s_trial3_2024.11.27-03.01.42_Step_322000_improve_rat_0.0753710768617964_imporve_num_260_yaml_name_run_train_sin_k8s_trial3.txt', file_without_CDR_weight='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test/refinefile_model_zoo_nb_no4_95_testset_0522_refinemodel_run_train_sin_k8s_trial3_yaml_run_train_sin_k8s_trial6_2024.12.16-09.24.27_Step_16000_improve_rmsd_-0.11802367705944156_imporve_num_211_yaml_name_run_train_sin_k8s_trial6.txt')
    # analyze_distribution_pdb_trans('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset_train.csv')
    
    #随机划分出10%作为validation 数据
    # 先获得所有的数据
    statics = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset_valid_statics.csv'
    
    df = load_data(statics)
    first_col_value = []
    for index, row in df.iterrows():
        first_col_value.append(row.iloc[0])  # row的第0列
    random.shuffle(first_col_value)
    train_data = first_col_value[:int(0.9*len(first_col_value))]
    validation_data = first_col_value[int(0.9*len(first_col_value)):]

    data_train_fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset_train_all_decoys.csv'
    data_validation_fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset_validation_one_decoys_each_model.csv'
    # 获得所有的数据
    all_data = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset_valid.csv'
    # 使用 .isin() 方法过滤 DataFrame
    # 定义每个块的行数
    # 打开输入和输出文件
    train_flag = 0
    with open(all_data, 'r', newline='', encoding='utf-8') as infile, \
        open(data_train_fpath, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 逐行读取并过滤
        for row in reader:
            
            if row['id'] in train_data:
                print(train_flag)
                train_flag += 1
                writer.writerow(row)


    train_flag = 0
    #加入一个list，去判断是否会出现相同model出来的decoys
    validation_s = []

    with open(all_data, 'r', newline='', encoding='utf-8') as infile, \
        open(data_validation_fpath, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        # 写入表头
        writer.writeheader()
        
        # 逐行读取并过滤
        for row in reader:
            # 以id加上fpath中以id分割的前段相加作为
            
            if row['id'] in validation_data:
                tem_chart = row['id']+row['pdb_fpath'].split(row['id'])[0]
                if tem_chart in validation_s:
                    continue
                else:
                    print(train_flag)
                    train_flag += 1
                    writer.writerow(row)
                    validation_s.append(tem_chart)


    


    # rmsd = calculate_ca_rmsd('/nfs_beijing_ai/jinxian/ABlooper_new/fv159_vh-vl-scoring-model-0-1-10-23-24-30-31-32-33-34-35-36-37-38-39-40-41-42-43-44-41-42-43-44-41-42-43-44_ensemble_output/7bh8_ABlooper.pdb', '/nfs_beijing_ai/jinxian/ABlooper/fv159_vh-vl-scoring-model-0-1-10-23-24-30-31-32-33-34-35-36-37-38-39-40-41-42-43-44-41-42-43-44-41-42-43-44_ensemble_output/7bh8_ABlooper.pdb')
    # inference_sort_pdb_by_mean_rmsd()
    # print(len('EVQLVESGGGLVQAGGSLRLSCAASGSDFSANAVGWYRQAPGKQRVVVASISSTGNTKYSNSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCWLFRFGIENYWGQGTQVTVSS'))
    # valid_sabdab('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SabDab/Sabdab_dataset/7b2m.pdb')
    # process_AFDB('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/AFDB_train_dataset.csv','/nfs_beijing_ai/jinxian/GNNRefine/data/target.lst')
    # sort_pdb_by_mean_rmsd()
    # sis_CA_rmsd('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/AFDB_train_dataset.csv')
    
    # main()
    # # 再筛选一次pdb。根据decoys的XXXX_X的形式，找到chain id，然后去原始gt pdb中拿到对应的chain。其中要先确认文件地址存在。
    # # 需要保留的是id,pdb,chain,pdb_fpath,pdb_fpath_gt,resolution,full_seq,res_idx,res_is_add,chain_type,full_seq_AMR,similarity
    # # decoys文件有四个，esmflow，modeller，openmm和torchMD。
    # # gt文件为/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/trainset_gt_22059_nb15_nb22_vh15_vh9_vl11_model_decoys_0423.csv
    # gt_file_path = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/trainset_gt_22059_nb15_nb22_vh15_vh9_vl11_model_decoys_0423.csv'
    # df = load_data(gt_file_path)
    # filtered_df_vhh = df[df['chain_type'] == 'vhh']
    # print(filtered_df_vhh[0:4])
    # unique_id = {}
    
    # filter_data = []
    # _7q1u_B_all = []
    # _7q1u_B_found = []

    # col_name = ['id','pdb','chain','pdb_fpath','pdb_fpath_gt','resolution','full_seq','res_idx','res_is_add','chain_type','full_seq_AMR','similarity']
    # num_flag = 0
    # esmflow_fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/nb_train_esmflow_2071_all_95_0510_filtered_w_pdbid.csv'
    # esmflow_df = load_data(esmflow_fpath)
    # esmflow_pred_path = list(esmflow_df["pred_path"])
    # esmflow_pdb_id =  list(esmflow_df["pdb_id"])
    # for i in range(len(esmflow_pdb_id)):
    #     #先判断pred file是否存在
    #     # if esmflow_pdb_id[i]=='7q1u_B':
    #     #     _7q1u_B_all.append(esmflow_pred_path[i])
            
    #     # if os.path.exists(esmflow_pred_path[i]):
    #     #     if esmflow_pdb_id[i]=='7q1u_B':
    #     #         _7q1u_B_found.append(esmflow_pred_path[i])

    #         #根据decoys的pdb id 找到对应的gt 中对应id。
    #         matched_row = filtered_df_vhh[filtered_df_vhh['id'] == esmflow_pdb_id[i]]

    #         gt_pdb_fpath = list(matched_row["pdb_fpath"])[0]
    #         gt_id = list(matched_row["id"])[0]
    #         # 如果gt pdb存在
    #         if os.path.exists(gt_pdb_fpath) and (gt_id == esmflow_pdb_id[i]):
                
    #             num_flag += 1

    #             gt_pdb = list(matched_row["pdb"])[0]
    #             gt_chain = list(matched_row["chain"])[0]
                
    #             gt_resolution = list(matched_row["resolution"])[0]
    #             gt_full_seq = list(matched_row["full_seq"])[0]
    #             gt_res_idx = list(matched_row["res_idx"])[0]
    #             gt_res_is_add = list(matched_row["res_is_add"])[0]
    #             gt_chain_type = list(matched_row["chain_type"])[0]
    #             gt_full_seq_AMR = list(matched_row["full_seq_AMR"])[0]
    #             gt_similarity = list(matched_row["similarity"])[0]
    #             filter_data.append([gt_id,gt_pdb,gt_chain,esmflow_pred_path[i],gt_pdb_fpath,gt_resolution,gt_full_seq,gt_res_idx,gt_res_is_add,gt_chain_type,gt_full_seq_AMR,gt_similarity])
    #             if gt_id in unique_id:
    #                 unique_id[gt_id] += 1
    #             else:
    #                 unique_id[gt_id] = 1
    # print(num_flag)

            
    # modeller_fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/nb_train_modeller_2071_all_95_0430_w_pdbid.csv'
    # modeller_df = load_data(modeller_fpath)
    # modeller_pred_path = list(modeller_df["pred_path"])
    # modeller_pdb_id =  list(modeller_df["pdb_id"])
    # for i in range(len(modeller_pdb_id)):

    #     if modeller_pdb_id[i]=='7q1u_B':
    #         _7q1u_B_all.append(modeller_pred_path[i])

    #     # 先判断pred file是否存在
    #     if os.path.exists(modeller_pred_path[i]):
    #         if modeller_pdb_id[i]=='7q1u_B':
    #             _7q1u_B_found.append(modeller_pred_path[i])
    #         matched_row = filtered_df_vhh[filtered_df_vhh['id'] == modeller_pdb_id[i]]

    #         gt_pdb_fpath = list(matched_row["pdb_fpath"])[0]
    #         gt_id = list(matched_row["id"])[0]
    #         # 如果gt pdb存在
    #         if os.path.exists(gt_pdb_fpath) and (gt_id == modeller_pdb_id[i]):
                
    #             num_flag += 1

    #             gt_pdb = list(matched_row["pdb"])[0]
    #             gt_chain = list(matched_row["chain"])[0]
                
    #             gt_resolution = list(matched_row["resolution"])[0]
    #             gt_full_seq = list(matched_row["full_seq"])[0]
    #             gt_res_idx = list(matched_row["res_idx"])[0]
    #             gt_res_is_add = list(matched_row["res_is_add"])[0]
    #             gt_chain_type = list(matched_row["chain_type"])[0]
    #             gt_full_seq_AMR = list(matched_row["full_seq_AMR"])[0]
    #             gt_similarity = list(matched_row["similarity"])[0]
    #             filter_data.append([gt_id,gt_pdb,gt_chain,modeller_pred_path[i],gt_pdb_fpath,gt_resolution,gt_full_seq,gt_res_idx,gt_res_is_add,gt_chain_type,gt_full_seq_AMR,gt_similarity])
    #             if gt_id in unique_id:
    #                 unique_id[gt_id] += 1
    #             else:
    #                 unique_id[gt_id] = 1
    # print(num_flag)


    # torchMD_fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/nb_train_torchMD_2071_all_95_0511_w_pdbid.csv'
    # torchMD_df = load_data(torchMD_fpath)
    # torchMD_pred_path = list(torchMD_df["pred_path"])
    # torchMD_pdb_id =  list(torchMD_df["pdb_id"])

    # for i in range(len(torchMD_pdb_id)):

    #     if torchMD_pdb_id[i]=='7q1u_B':
    #         _7q1u_B_all.append(torchMD_pred_path[i])

    #     # 先判断pred file是否存在
    #     if os.path.exists(torchMD_pred_path[i]):
    #         if torchMD_pdb_id[i]=='7q1u_B':
    #             _7q1u_B_found.append(torchMD_pred_path[i])
                
    #         matched_row = filtered_df_vhh[filtered_df_vhh['id'] == torchMD_pdb_id[i]]

    #         gt_pdb_fpath = list(matched_row["pdb_fpath"])[0]
    #         gt_id = list(matched_row["id"])[0]
    #         # 如果gt pdb存在
    #         if os.path.exists(gt_pdb_fpath) and (gt_id == torchMD_pdb_id[i]):
                
    #             num_flag += 1

    #             gt_pdb = list(matched_row["pdb"])[0]
    #             gt_chain = list(matched_row["chain"])[0]
                
    #             gt_resolution = list(matched_row["resolution"])[0]
    #             gt_full_seq = list(matched_row["full_seq"])[0]
    #             gt_res_idx = list(matched_row["res_idx"])[0]
    #             gt_res_is_add = list(matched_row["res_is_add"])[0]
    #             gt_chain_type = list(matched_row["chain_type"])[0]
    #             gt_full_seq_AMR = list(matched_row["full_seq_AMR"])[0]
    #             gt_similarity = list(matched_row["similarity"])[0]
    #             filter_data.append([gt_id,gt_pdb,gt_chain,torchMD_pred_path[i],gt_pdb_fpath,gt_resolution,gt_full_seq,gt_res_idx,gt_res_is_add,gt_chain_type,gt_full_seq_AMR,gt_similarity])
    #             if gt_id in unique_id:
    #                 unique_id[gt_id] += 1
    #             else:
    #                 unique_id[gt_id] = 1
    # print(num_flag)
    
    # # 定义 CSV 文件名
    # # csv_filename = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset_valid_statics.csv'

    # # # 打开文件进行写入
    # # with open(csv_filename, mode='w', newline='') as file:
    # #     writer = csv.writer(file)
        
    # #     # 写入每对键值对为一行
    # #     for key, value in unique_id.items():
    # #         writer.writerow([key, value])
    
    # df = pd.DataFrame(_7q1u_B_all)  # 替换'Column1'为你想要的列名
    
    # # 保存为CSV文件
    # df.to_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/7q1u_B_all.csv', index=False)
    # df = pd.DataFrame(_7q1u_B_found)  # 替换'Column1'为你想要的列名
    
    # # 保存为CSV文件
    # df.to_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/7q1u_B_found.csv', index=False)
