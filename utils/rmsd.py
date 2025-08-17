"""Code."""
# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from Bio import PDB
import re
from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np
import torch
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem, GetPeriodicTable, RemoveHs
from rdkit.Geometry import Point3D
from scipy import spatial
from scipy.special import softmax
from torch_cluster import radius_graph
from Bio.PDB.Polypeptide import is_aa
import os
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Bio import SeqIO, pairwise2

# from utils.utils import get_align_rotran


biopython_pdbparser = PDBParser(QUIET=True)
biopython_cifparser = MMCIFParser()
from Bio.PDB import PDBParser, Superimposer  
import numpy as np  

def get_ca_atoms(structure):
    ca_atoms = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    res_id = (chain.id, residue.id[1])
                    ca_atoms[res_id] = residue['CA'].coord
    return ca_atoms

def find_residue_index(seq, resname, start=0):
    try:
        return seq.index(resname, start)
    except ValueError:
        return -1

    

def align_proteins_and_calculate_rmsd(protein1_file, protein2_file):  
    # 创建PDB解析器  
    parser = PDBParser()  
  
    # 解析两个蛋白质结构  
    structure1 = parser.get_structure("protein1", protein1_file)  
    structure2 = parser.get_structure("protein2", protein2_file)
  
    # 获取Cα原子  
    ca_atoms1 = [atom for atom in structure1.get_atoms() if atom.get_name() == 'CA']  
    ca_atoms2 = [atom for atom in structure2.get_atoms() if atom.get_name() == 'CA']

    # 确保两个蛋白质有相同数量的Cα原子  
    if len(ca_atoms1) != len(ca_atoms2):  
        return False

    # if len(atoms1) != len(atoms2):  
    #     raise ValueError("两个蛋白质必须包含相同数量的主链原子。")   
  
    # 使用Superimposer进行对齐  
    superimposer = Superimposer()  
    superimposer.set_atoms(ca_atoms1, ca_atoms2)   
  
    ca_rmsd = superimposer.rms
       
  
    # rmsd = superimposer.rms
    # superimposer.rotran
    # aligned_rmsd = superimposer.rms
  
    return ca_rmsd 
  
# 使用示例
# df = pd.read_csv('/nfs_beijing_ai/jinxian/GNNRefine/data/output/casp15_af3_GT.csv')

# # 使用列名获取指定列的数据
# GT_file_paths = df['pdb_fpath'].tolist()

# file_path = '/nfs_beijing_ai/jinxian/GNNRefine/data/output/casp15_af3_Start.txt'

# # 读取txt文件内容，并保存为list
# with open(file_path, 'r', encoding='utf-8') as file:
#     lines = file.readlines()

# # 去掉每一行的换行符
# S_file_paths = [line.strip() for line in lines]

# # 读取refine txt文件内容，并保存为list
# with open('/nfs_beijing_ai/jinxian/GNNRefine/data/output/casp15_af3_Refine.txt', 'r', encoding='utf-8') as file:
#     lines = file.readlines()

# # 去掉每一行的换行符
# R_file_paths = [line.strip() for line in lines]

# rmsd_start = []
# rmsd_refine = []
# names = []
# flag = 0
# for i in range(len(GT_file_paths)):
#     protein_pdb = S_file_paths[i]
#     label = GT_file_paths[i] 
#     ca_rmsd_s = align_proteins_and_calculate_rmsd(label,protein_pdb)  
#     if ca_rmsd_s == False:
#         continue


#     re_label = R_file_paths[i]
#     ca_rmsd = align_proteins_and_calculate_rmsd(re_label,protein_pdb) 
#     if ca_rmsd == False:
#         continue
#     # show result
#     print('PDB: ', S_file_paths[i])
#     print(f"RMSD before alignment(CA): {ca_rmsd_s}")  
#     rmsd_start.append(ca_rmsd_s)

    
#     # show result
#     print(f"RMSD before alignment(CA): {ca_rmsd}")  
#     rmsd_refine.append(ca_rmsd)

#     names.append(GT_file_paths[i].split('/')[-1])
   

# 示例数据
# names = names
# values_before = rmsd_start
# values_after = rmsd_refine

# # 绘制条形图
# x = np.arange(len(names))  # 标签位置
# width = 0.35  # 条形图的宽度

# fig, ax = plt.subplots()
# bars_before = ax.bar(x - width/2, values_before, width, label='Before')
# bars_after = ax.bar(x + width/2, values_after, width, label='After')

# # 添加一些文本标签
# ax.set_xlabel('Data Names')
# ax.set_ylabel('Values')
# ax.set_title('Values Before and After')
# ax.set_xticks(x)
# ax.set_xticklabels(names)
# ax.legend()
# plt.xticks(rotation=90)

# fig.tight_layout()
# plt.savefig('/nfs_beijing_ai/jinxian/GNNRefine/data/output/before_after_comparison.png', format='png')
# plt.show()
# print('over')


def superimpose_single(reference, coords):
    """Run  superimpose_single method."""
    # code.
    """
    Superimposes coordinates onto a reference by minimizing RMSD using SVD.

    Args:
        reference:
            [N, 3] reference tensor
        coords:
            [N, 3] tensor
    Returns:
        A tuple of [N, 3] superimposed coords and the final RMSD.
    """

    # Convert to numpy if the input is a tensor
    if isinstance(reference, torch.Tensor):
        reference = reference.detach().cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()

    # Superimpose using SVD
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    rot, tran = sup.get_rotran()
    superimposed = sup.get_transformed()
    rmsd = sup.get_rms()

    # convert back to tensor
    rot = torch.from_numpy(rot)
    tran = torch.from_numpy(tran)
    superimposed = torch.from_numpy(superimposed)
    rmsd = torch.tensor(rmsd)
    return rot, tran, superimposed, rmsd


def select_unmasked_coords(coords, mask):
    """Run  select_unmasked_coords method."""
    # code.
    return torch.masked_select(
        coords,
        (mask > 0.0)[..., None],
    ).reshape(-1, 3)


def superimpose(reference, coords, mask):
    """Run  superimpose method."""
    # code.
    r, c, m = reference, coords, mask
    r_unmasked_coords = select_unmasked_coords(r, m)
    c_unmasked_coords = select_unmasked_coords(c, m)
    rot, tran, superimposed, rmsd  = superimpose_single(r_unmasked_coords, c_unmasked_coords)

    return rot, tran, rmsd, superimposed


def cdr_global_superimpose(reference, coords, mask, rot, tran):
    """Run  cdr_global_superimpose method."""
    # code.
    r, c, m = reference, coords, mask
    r_unmasked_coords = select_unmasked_coords(r, m)
    c_unmasked_coords = select_unmasked_coords(c, m)

    superimposed_global_align = (
        torch.mm(c_unmasked_coords, rot.detach()) + tran.detach()
    )
    rmsd_global_align = torch.sqrt(
        torch.mean(
            torch.sum(
                (superimposed_global_align - r_unmasked_coords) ** 2,
                dim=-1,
            )
        )
    )
    return rmsd_global_align



def get_pos_mask(structure, structure_af2):
    alpha_carbons = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    try:
                        alpha_carbon = residue["CA"]
                        alpha_carbons.append(alpha_carbon.get_coord())
                    except KeyError:
                        # Some residues might not have alpha carbon, skipping them
                        pass

    # 打印所有alpha碳原子的坐标
    # print(len(alpha_carbons))

    alpha_carbons_af2 = []
    for model in structure_af2:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    try:
                        alpha_carbon = residue["CA"]
                        alpha_carbons_af2.append(alpha_carbon.get_coord())
                    except KeyError:
                        # Some residues might not have alpha carbon, skipping them
                        pass
    # print(len(alpha_carbons_af2))

    if len(alpha_carbons) == len(alpha_carbons_af2):
        alpha_carbons = torch.tensor(alpha_carbons)
        alpha_carbons_af2 = torch.tensor(alpha_carbons_af2)
        list_of_ones = [1] * len(alpha_carbons)
        mask_list = torch.tensor(list_of_ones)
        return alpha_carbons, alpha_carbons_af2, mask_list, True
    else:
        return 0, 0, 0, False



# gt = torch.tensor([[1, 2, 3], [4, 5,6]])
# af = torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1,6.1]])
# mask_pos = torch.tensor([1, 1])

# rmsd = superimpose(gt, af, mask_pos)
# print(rmsd)
def get_protein(file_type, file_dir):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        if file_type == '.pdb':
            structure = biopython_pdbparser.get_structure('pdb', file_dir)
        elif file_type == '.cif':
            structure = biopython_cifparser.get_structure('cif', file_dir)
        else:
            raise "protein is not pdb or cif"
            return False
    return structure

def get_alpha_carbons(structure):
    
    alpha_carbons = []
    # ca_coordinates = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    try:
                        # res_id = residue.id[1]
                        # ca_coordinates[res_id] = residue['CA'].get_coord()

                        alpha_carbon = residue["CA"]
                        alpha_carbons.append(alpha_carbon.get_coord())
                        
                    except KeyError:
                        # Some residues might not have alpha carbon, skipping them
                        pass

    return torch.tensor(np.array(alpha_carbons))

# 加载PDB文件

# alpha_carbons = get_alpha_carbons('.pdb', '/nfs_beijing_ai/jinxian/GNNRefine/data/A0QND6/5ah2_aligned_to_A0QND6.pdb')

# alpha_carbons_af2 = get_alpha_carbons('.pdb', '/nfs_beijing_ai/jinxian/GNNRefine/data/A0QND6/af2_5ah2_aligned.pdb')

# alpha_carbons_refine = get_alpha_carbons('.pdb', '/nfs_beijing_ai/jinxian/GNNRefine/data/output/af2_5ah2_aligned.pdb.refined.pdb')


# list_of_ones = [1] * len(alpha_carbons)
# mask_list = torch.tensor(list_of_ones)
# print(len(mask_list))
# rot, tran, superimposed, rmsd =  superimpose(alpha_carbons, alpha_carbons_af2, mask_list)
# print("GT-AF rmsd:", rmsd)
# rot, tran, superimposed, rmsd =  superimpose(alpha_carbons, alpha_carbons_refine, mask_list)
# print("GT-Refine rmsd:", rmsd)
# rot, tran, superimposed, rmsd =  superimpose(alpha_carbons_refine, alpha_carbons_af2, mask_list)
# print("Refine-AF rmsd:", rmsd)
def get_sequence(structure):
    """Get the amino acid sequence of a PDB structure."""
    ppb = PDB.PPBuilder()
    seq = ""
    for pp in ppb.build_peptides(structure):
        seq += pp.get_sequence()
    return seq

def find_matching_domain(target_seq, source_structure):
    """Find the matching domain in the source structure that matches the target sequence."""
    ppb = PDB.PPBuilder()
    for pp in ppb.build_peptides(source_structure):
        seq = pp.get_sequence()
        if target_seq in seq:
            return pp
    return None

def extract_domain(residues):
    """Extract the domain from residues."""
    chain = PDB.Chain.Chain('A')
    for res in residues:
        chain.add(res.copy())
    
    model = PDB.Model.Model(0)
    model.add(chain)
    
    new_structure = PDB.Structure.Structure('domain')
    new_structure.add(model)
    return new_structure


def match_protein(target_structure, source_structure):
    # Get the sequence of the target structure
    target_seq = get_sequence(target_structure)
    

    # Find the matching domain in the source structure
    matching_domain = find_matching_domain(target_seq, source_structure)
    if matching_domain is None:
        print("No matching domain found.")
        return False

    # Extract the matching residues
    matching_residues = [res for res in matching_domain]
    
    # Extract the domain
    new_structure = extract_domain(matching_residues)

    # return
    return new_structure

def gdt_cal(reference, model, thresholds):
    distances = np.sqrt(np.sum((reference - model) ** 2, axis=1))
    scores = [(distances < t).sum() / len(distances) for t in thresholds]
    return np.mean(scores)

def gdt(reference, model):
    return gdt_cal(reference.numpy(), model.numpy(), [1.0, 2.0, 4.0, 8.0]), gdt_cal(reference.numpy(), model.numpy(), [0.5, 1.0, 2.0, 4.0])


def lddt(reference, model, cutoff=15.0):
    def local_lddt(ref, pred, cutoff):
        ref_distances = np.sqrt(np.sum((ref[:, None] - ref[None, :]) ** 2, axis=2))
        pred_distances = np.sqrt(np.sum((pred[:, None] - pred[None, :]) ** 2, axis=2))
        mask = (ref_distances < cutoff) & (ref_distances > 0)
        diff = np.abs(ref_distances - pred_distances)
        score = ((diff < 0.5).sum() + (diff < 1.0).sum() + (diff < 2.0).sum() + (diff < 4.0).sum()) / 4
        return score / mask.sum()

    local_scores = [local_lddt(reference.numpy(), model.numpy(), cutoff) for i in range(len(reference.numpy()))]
    global_lddt = np.mean(local_scores)
    return global_lddt, local_scores
    
# GT_dir = '/nfs_beijing_ai/jinxian/GNNRefine/data/casp13.targets.R.4predictors'
# start_dir = '/nfs_beijing_ai/jinxian/GNNRefine/data/0401_scratch_rmsd_weighted'

# file_name = 'casp14_af3'
# Refine_dir = '/nfs_beijing_ai/jinxian/GNNRefine/data/output/'+ file_name + '_Refine'

# Refine_file_paths = os.listdir(Refine_dir)
# print('Refine_file_paths length is ', len(Refine_file_paths))
# print(Refine_file_paths)

# # GT_file_paths = os.listdir(GT_dir)
# # Start_file_paths = os.listdir(start_dir)
# # 指定CSV文件路径
# file_path = '/nfs_beijing_ai/jinxian/remote/ProRefine/data/casp15_gt.csv'
# # 指定列名
# column_name = 'pdb_fpath'

# # 使用pandas读取CSV文件
# df = pd.read_csv(file_path)

# # 使用列名获取指定列的数据
# GT_file_paths = df[column_name].tolist()
# print('GT_file_paths length is ', len(GT_file_paths))
# print(GT_file_paths)

# GT_full_seq_AMR = df['full_seq_AMR'].tolist()

# 用新的path
# start_pdbs = []
# Start_file_paths = os.listdir('/nfs_beijing_ai/jinxian/GNNRefine/data/'+file_name)
# for pro in Start_file_paths:
#     if pro[-3:] == 'log':
#         continue
#     pro_file_path = os.listdir('/nfs_beijing_ai/jinxian/GNNRefine/data/' + file_name +'/'+pro)
    
#     for pdb_file in pro_file_path:
#         if (pdb_file[-4:] == '.pdb' or  pdb_file[-4:] == '.cif'):
#             start_pdbs.append('/nfs_beijing_ai/jinxian/GNNRefine/data/'+file_name+'/'+pro+'/'+pdb_file)
# Start_file_paths = start_pdbs

# 用新的path
    # start_pdbs = [] casp15_af3
# file_path = '/nfs_beijing_ai/jinxian/GNNRefine/data/pred_path/pred_path/' + file_name + '.txt'

# # 初始化一个空列表，用于存储文件中的数据
# Start_file_paths = []

# # 使用with语句打开文件，确保文件会在操作完成后自动关闭
# with open(file_path, 'r', encoding='utf-8') as file:
#     # 逐行读取文件
#     for line in file:
#         # 使用rstrip()移除行尾的换行符和其他空白字符
#         clean_line = line.rstrip()
#         # 将清理后的数据添加到列表中
#         Start_file_paths.append(clean_line)

# print('Start_file_paths length is ', len(Start_file_paths))
# print(Start_file_paths)





#筛选出Start model和GT model相同的pair
# complete_dir_GT = []
# complete_dir_Start = []
# complete_dir_Refine = []
# complete_GT_full_seq_AMR = []
# pattern = re.compile(r'casp15/(.*?\.pdb)')

# for i, GT_file_dir in enumerate(GT_file_paths):
#     match = pattern.search(GT_file_dir)
#     if match:
#         extracted_string = match.group(1).replace('.pdb', '')
#         GT_name = extracted_string.replace('-', '_')
#         print('get name:', GT_name)
#     else:
#         continue
    
#     for Refine_file_dir in Refine_file_paths:
#         #如果匹配上
#         # print(start_name)
#         for i, Start_file_dir in enumerate(Start_file_paths):
#             if (GT_name.lower()  in Start_file_dir.lower() and GT_name.lower()  in Refine_file_dir.lower()):
#                 s_ca, _ = get_alpha_carbons(get_protein('.pdb', Start_file_dir))
#                 r_ca, _ = get_alpha_carbons(get_protein('.pdb', '/nfs_beijing_ai/jinxian/GNNRefine/data/output/'+ file_name + '_Refine/'+ Refine_file_dir))
#                 if len(s_ca) != len(r_ca):
#                     break

#                 #将pair的路径存储起来
#                 complete_dir_GT.append(GT_file_dir)
#                 complete_GT_full_seq_AMR.append(GT_full_seq_AMR[i])
#                 complete_dir_Start.append(Start_file_dir)
#                 complete_dir_Refine.append('/nfs_beijing_ai/jinxian/GNNRefine/data/output/'+ file_name + '_Refine/'+ Refine_file_dir)
#                 break
    


# #计算refine之前和之后的RMSD以及其他评价指标
# #Before refine 
# num_protein = len(complete_dir_GT)
# print('the num of matching data: ', num_protein)

# import csv

# # 存储GT
# list1 = complete_GT_full_seq_AMR
# list2 = complete_dir_GT

# # 结合两个列表为一个列表的元组
# combined_list = list(zip(list1, list2))

# # 指定CSV文件路径
# csv_file = '/nfs_beijing_ai/jinxian/GNNRefine/data/output/'+ file_name+'_GT.csv'

# # 写入CSV文件
# with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(['full_seq_AMR', 'pdb_fpath'])  # 写入表头
#     writer.writerows(combined_list)  # 写入数据

# # 存储Start
# my_list = complete_dir_Start
# txt_file = '/nfs_beijing_ai/jinxian/GNNRefine/data/output/'+ file_name+'_Start.txt'

# # 打开文件准备写入
# with open(txt_file, 'w', encoding='utf-8') as file:
#     # 遍历列表并写入每一行
#     for item in my_list:
#         # 写入每个元素，元素之间用换行符分隔
#         file.write(f"{item}\n")

# # 存储Refine
# my_list = complete_dir_Refine
# txt_file = '/nfs_beijing_ai/jinxian/GNNRefine/data/output/'+ file_name+'_Refine.txt'

# # 打开文件准备写入
# with open(txt_file, 'w', encoding='utf-8') as file:
#     # 遍历列表并写入每一行
#     for item in my_list:
#         # 写入每个元素，元素之间用换行符分隔
#         file.write(f"{item}\n")



# sta_before_RMSD = []
# sta_before_gdt_ts = []
# sta_before_gdt_ha = []
# sta_before_global_lddt = []

# sta_after_RMSD = []
# sta_after_gdt_ts = []
# sta_after_gdt_ha = []
# sta_after_global_lddt = []

# num_refine = 0

# for i in range(num_protein):
#     GT_path = complete_dir_GT[i]
#     Start_path = complete_dir_Start[i]
#     GT_protein = get_protein(GT_path[-4:],GT_path)
#     Start_protein = get_protein(Start_path[-4:],Start_path)
#     # new_GT_proten = match_protein(Start_protein, GT_protein)
#     # if new_GT_proten == False:
#     #     continue

#     Start_alpha_carbons, Start_ca_coordinates = get_alpha_carbons(Start_protein)
#     # GT_alpha_carbons = get_alpha_carbons(get_protein(GT_path[-4:],GT_path))
#     GT_alpha_carbons, GT_ca_coordinates = get_alpha_carbons(GT_protein)

#     Refine_path = complete_dir_Refine[i]
#     Refine_protein = get_protein(Refine_path[-4:],Refine_path)
#     Refine_alpha_carbons, Refine_ca_coordinates = get_alpha_carbons(Refine_protein)


#     print('computer the ', str(i) + '-th data')
#     print('Protein GT_path is : ', GT_path)
#     print('Protein Start_path is : ', Start_path)
#     print('Protein Refine_path is : ', Refine_path)

#     print('before matching!')
#     print('the length of GT_alpha_carbons:', len(GT_alpha_carbons))
#     print('the length of Start_alpha_carbons:', len(Start_alpha_carbons))
#     print('the length of Refine_alpha_carbons:', len(Refine_alpha_carbons))

#     # 找出共同的残基
#     common_residues = set(GT_ca_coordinates.keys()).intersection(set(Refine_ca_coordinates.keys())).intersection(set(Start_ca_coordinates.keys()))

#     if not common_residues:
#         raise ValueError("No common residues found between the two PDB files.")

#     # 提取共同残基的Cα坐标
#     GT_alpha_carbons = torch.tensor(np.array([GT_ca_coordinates[res] for res in sorted(common_residues)]))
#     Start_alpha_carbons = torch.tensor(np.array([Start_ca_coordinates[res] for res in sorted(common_residues)]))
#     Refine_alpha_carbons = torch.tensor(np.array([Refine_ca_coordinates[res] for res in sorted(common_residues)]))
#     print('after matching!')
#     print('the length of GT_alpha_carbons:', len(GT_alpha_carbons))
#     print('the length of Start_alpha_carbons:', len(Start_alpha_carbons))
#     print('the length of Refine_alpha_carbons:', len(Refine_alpha_carbons))
    
    

#     if len(GT_alpha_carbons) == len(Start_alpha_carbons) and len(Start_alpha_carbons) == len(Refine_alpha_carbons):
#         list_of_ones = [1] * len(GT_alpha_carbons)
#         mask_list = torch.tensor(list_of_ones)
#         before_rot, before_tran, before_superimposed, before_rmsd = superimpose(GT_alpha_carbons, Start_alpha_carbons, mask_list)
#         # before_gdt_ts, before_gdt_ha = gdt(GT_alpha_carbons, Start_alpha_carbons)
#         # before_global_lddt, before_lddt_scores = lddt(GT_alpha_carbons, Start_alpha_carbons)
        
#         after_rot, after_tran, after_superimposed, after_rmsd = superimpose(GT_alpha_carbons, Refine_alpha_carbons, mask_list)
#         # after_gdt_ts, after_gdt_ha = gdt(GT_alpha_carbons, Refine_alpha_carbons)
#         # after_global_lddt, after_lddt_scores = lddt(GT_alpha_carbons, Refine_alpha_carbons)
        
        

#         from scipy.spatial.distance import pdist, squareform
#         def calculate_distance_differences(real_atoms, predicted_atoms):
#             real_dists = squareform(pdist(real_atoms, 'euclidean'))
#             predicted_dists = squareform(pdist(predicted_atoms, 'euclidean'))
#             return real_dists, predicted_dists

#         # 计算GDT-TS和GDT-HA
#         def calculate_gdt_scores(real_dists, predicted_dists):
#             distance_differences = np.abs(real_dists - predicted_dists)
#             gdt_values = 1 / (1 + distance_differences)
#             gdt_ts = np.sum(gdt_values)
#             gdt_ha = gdt_ts / real_dists.shape[0]  # 假设是对称矩阵，所以除以残基数
#             return gdt_ts, gdt_ha

#         # 计算Global lDDT
#         def calculate_global_lDDT(real_dists, predicted_dists):
#             lddt = np.max(np.abs(real_dists - predicted_dists), axis=1)
#             global_lDDT = np.mean(lddt)
#             return global_lDDT

#         # 计算before距离差异
#         real_ca_atoms = np.array(GT_alpha_carbons) 
#         predicted_ca_atoms = np.array(Start_alpha_carbons)  # 预测的Cα原子坐标
#         real_dists, predicted_dists = calculate_distance_differences(real_ca_atoms, predicted_ca_atoms)

#         # 计算GDT-TS和GDT-HA
#         before_gdt_ts, before_gdt_ha = calculate_gdt_scores(real_dists, predicted_dists)

#         # 计算Global lDDT
#         before_global_lddt = calculate_global_lDDT(real_dists, predicted_dists)

#         # 计算after距离差异
#         real_ca_atoms = np.array(GT_alpha_carbons) 
#         predicted_ca_atoms = np.array(Refine_alpha_carbons)  # 预测的Cα原子坐标
#         real_dists, predicted_dists = calculate_distance_differences(real_ca_atoms, predicted_ca_atoms)

#         # 计算GDT-TS和GDT-HA
#         after_gdt_ts, after_gdt_ha = calculate_gdt_scores(real_dists, predicted_dists)

#         # 计算Global lDDT
#         after_global_lddt = calculate_global_lDDT(real_dists, predicted_dists)


        
#         print(f"Before RMSD: {before_rmsd}")
#         sta_before_RMSD.append(before_rmsd)
#         print(f"After RMSD: {after_rmsd}")
#         sta_after_RMSD.append(after_rmsd)

#         # print(f"Before GDT-TS score: {before_gdt_ts}")
#         # sta_before_gdt_ts.append(before_gdt_ts)
#         # print(f"After GDT-TS score: {after_gdt_ts}")
#         # sta_after_gdt_ts.append(after_gdt_ts)

#         print(f"Before GDT-HA score: {before_gdt_ha}")
#         sta_before_gdt_ha.append(before_gdt_ha)
#         print(f"After GDT-HA score: {after_gdt_ha}")
#         sta_after_gdt_ha.append(after_gdt_ha)
#         if after_rmsd > before_rmsd:
#             num_refine += 1


#         print(f"Before Global lDDT score: {before_global_lddt}")
#         sta_before_global_lddt.append(before_global_lddt)
#         print(f"After Global lDDT score: {after_global_lddt}")
#         sta_after_global_lddt.append(after_global_lddt)

#         # print(f"Before Local lDDT scores: {before_lddt_scores}")
#         # print(f"After Local lDDT scores: {after_lddt_scores}")
#         print('------------------------------')

#         # 计算差值
#         # dis_rmsd = after_rmsd-

#     else:
#         print('Protein is : ', GT_path)
#         print('长度不一致！！！')
#         print('GT length is :', len(GT_alpha_carbons) )
#         print('Start length is :', len(Start_alpha_carbons))
#         print('Refine length is :', len(Refine_alpha_carbons) )
#         print('---------------------')

#         continue


# print(f"Mean of Before RMSD: {np.mean(sta_before_RMSD)}")
# print(f"Mean of After RMSD: {np.mean(sta_after_RMSD)}")

# # print(f"Mean of Before GDT-TS score: {np.mean(sta_before_gdt_ts)}")
# # print(f"Mean of After GDT-TS score: {np.mean(sta_after_gdt_ts)}")

# print(f"Mean of Before GDT-HA score: {np.mean(sta_before_gdt_ha)}")
# print(f"Mean of After GDT-HA score: {np.mean(sta_after_gdt_ha)}")

# print(f"Mean of Before Global lDDT score: {np.mean(sta_before_global_lddt)}")
# print(f"Mean of After Global lDDT score: {np.mean(sta_after_global_lddt)}")
# print('we test ', len(complete_dir_GT), 'proteins')
# print('there have ', num_refine, 'better refinement')
# # pdbbind_dir = "/nfs_beijing_ai/jinxian/DynamicBind/mnt/nas/research-data/luwei/dynamicbind_data/pdbbind_v11/pocket_aligned_fill_missing/"
# # info=pd.read_csv('/nfs_beijing_ai/jinxian/DynamicBind/data/d3_with_clash_info.csv')
# # uid = info['uid']

# # rmsd_list = []
# # for pdbid in uid:
# #     print(pdbid)
# #     file_paths = os.listdir(os.path.join(pdbbind_dir, pdbid))
# #     crystal_rec_path = os.path.join(pdbbind_dir, pdbid, [path for path in file_paths if '_aligned_to_' in path][0])
# #     af2_rec_path = os.path.join(pdbbind_dir, pdbid, [path for path in file_paths if 'af2_' in path][0])

# #     structure = biopython_pdbparser.get_structure('pdb', crystal_rec_path)
# #     structure_af2 = biopython_pdbparser.get_structure('pdb', af2_rec_path)


# #     alpha_carbons, alpha_carbons_af2, mask_list, t = get_pos_mask(structure, structure_af2)
# #     if t == False:
# #         continue
# #     rot, tran, superimposed, rmsd =  superimpose(alpha_carbons, alpha_carbons_af2, mask_list)
# #     print(rmsd)
# #     rmsd_list.append(rmsd.numpy())


# # # 计算均值和中位数
# # data = rmsd_list
# # print(len(rmsd_list))
# # mean_value = np.mean(data)
# # median_value = np.median(data)

# # # 绘制分布图
# # plt.figure(figsize=(8, 6))
# # plt.hist(data, bins=10, color='skyblue', edgecolor='black')
# # plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_value:.2f}')
# # plt.axvline(median_value, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_value}')
# # plt.title('Distribution of Data')
# # plt.xlabel('Value')
# # plt.ylabel('Frequency')
# # plt.legend()
# # plt.show()
# # plt.savefig('/nfs_beijing_ai/jinxian/DynamicBind/output.png', dpi=300, bbox_inches='tight')
# # plt.close()
# # print(f"Mean: {mean_value}")
# # print(f"Median: {median_value}")

