"""Code."""
import os
import random

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import networkx as nx

from np.protein import from_pdb_string
import np.residue_constants as rc
from dataset.feature.featurizer import process
from utils.data_encoding import encode_structure, encode_features, extract_topology
from dataset.screener import UpperboundScreener, ProbabilityScreener1d
from utils.logger import Logger
from utils.constants.atom_constants import *
from utils.constants.residue_constants import *
from utils.tool_metrics.protein.feature import get_seq_info

logger = Logger.logger

from Bio import Align
import argparse

def get_args():
    """Run get_args method."""
    # code.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/model_zoo_nb_no4_95_testset_0522.csv",
        help="path of coarse graining",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/test/",
        help="path of coarse graining",
    )
    
    parser.add_argument(
        "--target_selected_sample_num",
        type=int,
        default=50,
        help="target_selected_sample_num",
    )

    parser.add_argument(
        "--initially_tried_seq_num",
        type=int,
        default=25,
        help="initially_tried_seq_num",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="seed",
    )
    args = parser.parse_args()

    return args

def alignment_score(str1, str2):
    """
    get similarity score
        :param str1: query sequence
        :param str2: input sequence
        :return: similarity score between 0 and 1
    """
    aligner = Align.PairwiseAligner()
    alignments = aligner.align(str1, str2)
    alignment = next(alignments)
    score = alignment.score

    return score / max(len(str1), len(str2))


def reorg_by_unique_seq(raw_dataset, gt_dataset):
    dataset_by_pdbid = raw_dataset.groupby("id")

    uniseq2pdbid_dict = {}
    for name, group in dataset_by_pdbid:
        gt_info = gt_dataset[gt_dataset['id'] == name].reset_index()
        assert gt_info['chain_type'][0] == 'vhh'
        seq_amr = gt_info['full_seq_AMR'][0]
        cleared_seq, _ = get_seq_info(seq_amr)
        if cleared_seq not in uniseq2pdbid_dict.keys():
            uniseq2pdbid_dict.update({cleared_seq: []})

        uniseq2pdbid_dict[cleared_seq].append(name)

    return uniseq2pdbid_dict


class Phase1Splitter(object):
    def __init__(self, args):
        super(Phase1Splitter, self).__init__()
        self.args = args
        self.target_selected_sample_num = args.target_selected_sample_num
        self.initially_tried_seq_num = args.initially_tried_seq_num


    def split(self, raw_dataset, gt_dataset):
        uniseq2pdbid_dict = reorg_by_unique_seq(raw_dataset, gt_dataset)
        self.uniseq2pdbid_dict = uniseq2pdbid_dict
        logger.info(f"Total number of unique sequences:{len(uniseq2pdbid_dict.keys())}")

        # As the unique sequences are not so many, we can directly compute the similarity between each of them
        total_seq_num = len(uniseq2pdbid_dict.keys())
        uniseqs = list(uniseq2pdbid_dict.keys())
        self.uniseqs = uniseqs
        sim_mat = np.zeros([total_seq_num, total_seq_num])
        for i in tqdm(range(total_seq_num)):
            sim_mat[i,i] = 1.0
            for j in range(i+1, total_seq_num):
                score = alignment_score(uniseqs[i],uniseqs[j])
                sim_mat[i, j] = score
                sim_mat[j, i] = score

        adj_mat = np.where(sim_mat > 0.9, 1, 0)
        G = nx.from_numpy_array(adj_mat)
        components = list(nx.connected_components(G))

        size_1_components, size_2_components, size_over3_components = self.reorg_connected_components_by_size(components)

        random.seed(self.args.seed)
        random.shuffle(size_1_components)
        random.shuffle(size_2_components)
        random.shuffle(size_over3_components)

        logger.info(f"Trying to select {self.initially_tried_seq_num} sequences for each subset to achieve sample num {self.target_selected_sample_num}.")
        validset_idx, testset_idx = self.split_set(size_1_components, size_2_components, size_over3_components)
        selected_num = self.cnt_smp_num_of_given_seq_list(validset_idx) + self.cnt_smp_num_of_given_seq_list(testset_idx)

        while selected_num < self.target_selected_sample_num:
            logger.info(f"Current sample num:{selected_num}, which is lower than the target {self.target_selected_sample_num}. Retry")
            self.initially_tried_seq_num += 1
            validset_idx, testset_idx = self.split_set(size_1_components, size_2_components, size_over3_components)
            selected_num = self.cnt_smp_num_of_given_seq_list(validset_idx) + self.cnt_smp_num_of_given_seq_list(
                testset_idx)

        logger.info(f"final:")
        logger.info(selected_num)
        logger.info(validset_idx)
        logger.info(testset_idx)

        validset_pdbs, testset_pdbs = self.merge_set(validset_idx, testset_idx)
        #logger.info(len(validset_pdbs) + len(testset_pdbs))
        logger.info(f"valid pdbs:{validset_pdbs}")
        logger.info(f"test pdbs:{testset_pdbs}")

        validset_df, testset_df, remainedset_df = self.pdbset_to_df(raw_dataset, validset_pdbs, testset_pdbs)

        return validset_df, testset_df, remainedset_df


    def pdbset_to_df(self, raw_dataset, validset_pdbs, testset_pdbs):
        validset_selected_df = raw_dataset[raw_dataset['id'].isin(validset_pdbs)].reset_index(drop=True)
        remained_df = raw_dataset[~raw_dataset['id'].isin(validset_pdbs)]
        testset_selected_df = remained_df[remained_df['id'].isin(testset_pdbs)].reset_index(drop=True)
        remained_df = remained_df[~remained_df['id'].isin(testset_pdbs)].reset_index(drop=True)
        print(validset_selected_df)
        print(testset_selected_df)
        print(remained_df)
        print(len(raw_dataset))
        print(len(validset_selected_df))
        print(len(testset_selected_df))
        print(len(remained_df))
        return validset_selected_df, testset_selected_df, remained_df


    def merge_set(self, validset_idx, testset_idx):
        validset_seq = []
        testset_seq = []
        for idx in validset_idx:
            validset_seq.append(self.uniseqs[idx])
        for idx in testset_idx:
            testset_seq.append(self.uniseqs[idx])

        validset_pdbs, testset_pdbs = [], []
        for seq in validset_seq:
            validset_pdbs.extend(self.uniseq2pdbid_dict[seq])
        for seq in testset_seq:
            testset_pdbs.extend(self.uniseq2pdbid_dict[seq])

        return validset_pdbs, testset_pdbs


    def split_set(self, size_1_components, size_2_components, size_over3_components):
        # Split the components to valid/test set.
        validset_seqs = []
        testset_seqs = []

        target_set = validset_seqs
        target_cnt = 0
        for component in size_1_components:
            target_cnt += len(component)
            target_set.extend(component)
            if target_cnt >= self.initially_tried_seq_num:
                target_set = testset_seqs
                target_cnt = 0

            if len(testset_seqs) >= self.initially_tried_seq_num:
                break

        if len(testset_seqs) < self.initially_tried_seq_num:
            target_cnt = len(testset_seqs)
            for component in size_2_components:
                target_set.extend(component)
                target_cnt += len(component)

                if target_cnt >= self.initially_tried_seq_num:
                    break

        if len(testset_seqs) < self.initially_tried_seq_num:
            target_cnt = len(testset_seqs)
            for component in size_over3_components:
                target_set.extend(component)
                target_cnt += len(component)

                if target_cnt >= self.initially_tried_seq_num:
                    break

        return validset_seqs, testset_seqs


    def reorg_connected_components_by_size(self, components):
        size_1_components = []
        size_2_components = []
        size_over3_components = []

        for component in components:
            if len(component) == 1:
                size_1_components.append(component)
            elif len(component) == 2:
                size_2_components.append(component)
            else:
                size_over3_components.append(component)

        return size_1_components, size_2_components, size_over3_components

    def cnt_smp_num_of_given_seq_list(self, seq_list):
        smp_cnt = 0
        for seq in seq_list:
            if seq.__class__ == str:
                smp_cnt += len(self.uniseq2pdbid_dict[seq])
            else:
                seq = self.uniseqs[seq]
                smp_cnt += len(self.uniseq2pdbid_dict[seq])
        return smp_cnt

    def check_similarity_of_two_seq_lists(self, selected_seq_list, remained_seq_list):
        poped_seq = []
        no_poped_seq = []
        for i in range(len(selected_seq_list)):
            valid_seq = selected_seq_list[i]
            conflict_flag = False

            for j in range(len(remained_seq_list)):
                train_seq = remained_seq_list[j]
                align_score = alignment_score(valid_seq, train_seq)

                if align_score > 0.9:
                    print("similar seq detected.")
                    print(f"valid seq:{valid_seq}")
                    print(f"train seq:{train_seq}")
                    print(f"alignscore:{align_score}")
                    conflict_flag = True
                    continue

            if conflict_flag:
                poped_seq.append(valid_seq)
            else:
                no_poped_seq.append(valid_seq)

        print(len(poped_seq))
        print(len(no_poped_seq))
        return poped_seq, no_poped_seq


if __name__ == '__main__':
    #根据蛋白质序列的相似性划分训练和验证集
    #先获得gt蛋白和其序列
    #分别从static和train数据集中获得
    # statics_fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/NB_train_dataset_valid_statics.csv'
    # trainset_fpath = '/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/trainset_gt_22059_nb15_nb22_vh15_vh9_vl11_model_decoys_0423.csv'
    args = get_args()
    splitter = Phase1Splitter(args)
    #pandas读取后，再输入到split中
    raw_dataset = pd.read_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SAbDab_AB/AB_No_miss_res_fasta.csv')
    gt_dataset = pd.read_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SAbDab_AB/AB_No_miss_res_fasta.csv')
    validset_df, testset_df, remainedset_df = splitter.split(raw_dataset=raw_dataset, gt_dataset=gt_dataset)
    validset_df.to_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SAbDab_AB/AB_No_miss_res_fasta_vaild9.csv', index=False)
    testset_df.to_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SAbDab_AB/AB_No_miss_res_fasta_test9.csv', index=False)
    remainedset_df.to_csv('/nfs_beijing_ai/jinxian/rama-scoring1.3.0/dataset/datasets/SAbDab_AB/AB_No_miss_res_fasta_train9.csv', index=False)



# class Phase2Splitter(object):
#     def __init__(self, args):
#         super(Phase2Splitter, self).__init__()
#         self.args = args
#         self.seed = self.args.seed
#         self.num_decoy_per_sample = self.args.num_decoy_per_sample
#         self.phase2_sim_tolerance = self.args.phase2_sim_tolerance


#     def split(self, raw_dataset):
#         dataset_by_pdbid = raw_dataset.groupby("pdb_id")

#         validset_dfs = []
#         testset_dfs = []
#         remained_dfs = []
#         for name, group in tqdm(dataset_by_pdbid):
#             # print(f"processing {name}")
#             validset_df, testset_df, remained_df = self.split_decoys(group, name)
#             validset_dfs.append(validset_df)
#             testset_dfs.append(testset_df)
#             remained_dfs.append(remained_df)

#         print(f"len validset_dfs: {len(validset_dfs)}")
#         print(f"len testset_dfs: {len(testset_dfs)}")
#         validset_merged_df = pd.concat(validset_dfs, ignore_index=True)
#         testset_merged_df = pd.concat(testset_dfs, ignore_index=True)
#         remained_merged_df = pd.concat(remained_dfs, ignore_index=True)

#         print(len(validset_merged_df))
#         print(len(testset_merged_df))
#         print(len(remained_merged_df))

#         return validset_merged_df, testset_merged_df, remained_merged_df


#     def split_decoys(self, group_df, name):
#         group_df = group_df.reset_index(drop=True)
#         # print(group_df)
#         dc_num = len(group_df)
#         dc_sim_mat = np.zeros([dc_num, dc_num])
#         for i in range(dc_num):
#             info_i = [group_df['RMSD_CA'][i], group_df['CDR3_RMSD_CA'][i], group_df['rmsd_ca_95'][i], group_df['CDR3_RMSD_CA_95'][i]]
#             dc_sim_mat[i][i] = 1
#             for j in range(i+1, dc_num):
#                 info_j = [group_df['RMSD_CA'][j], group_df['CDR3_RMSD_CA'][j], group_df['rmsd_ca_95'][j],
#                           group_df['CDR3_RMSD_CA_95'][j]]

#                 if (abs(info_i[0] - info_j[0]) < self.phase2_sim_tolerance) and \
#                         (abs(info_i[1] - info_j[1]) < self.phase2_sim_tolerance) and \
#                         (abs(info_i[2] - info_j[2]) < self.phase2_sim_tolerance) and \
#                         (abs(info_i[3] - info_j[3]) < self.phase2_sim_tolerance):
#                     dc_sim_mat[i][j] = 1
#                     dc_sim_mat[j][i] = 1
#                 else:
#                     dc_sim_mat[i][j] = 0
#                     dc_sim_mat[j][i] = 0

#         G = nx.from_numpy_array(dc_sim_mat)
#         components = list(nx.connected_components(G))

#         size_x_components, size_over_x_components = self.reorg_connected_components_by_size(components)

#         random.seed(self.args.seed)
#         for size_i_components in size_x_components:
#             random.shuffle(size_i_components)
#         random.shuffle(size_over_x_components)

#         validset_idx, testset_idx = self.split_set(size_x_components, size_over_x_components, name)

#         validset_df = group_df.iloc[validset_idx].reset_index(drop=True)
#         testset_df = group_df.iloc[testset_idx].reset_index(drop=True)
#         remained_df = group_df.drop(index=validset_idx+testset_idx)
#         return validset_df, testset_df, remained_df


#     def reorg_connected_components_by_size(self, components):
#         size_x_components = []
#         for i in range(self.num_decoy_per_sample+1):
#             size_x_components.append([])
#         size_over_x_components = []

#         for component in components:
#             i = len(component)
#             if i <= self.num_decoy_per_sample+1:
#                 size_x_components[i-1].append(component)
#             else:
#                 size_over_x_components.append(component)


#         return size_x_components, size_over_x_components


#     def split_set(self, size_x_components, size_over_x_components, name):
#         # Split the components to valid/test set.
#         validset_decoys = []
#         testset_decoys = []

#         for size_i_components in size_x_components:
#             for component in size_i_components:
#                 if len(validset_decoys) < self.num_decoy_per_sample:
#                     validset_decoys.extend(component)
#                 elif len(testset_decoys) < self.num_decoy_per_sample:
#                         testset_decoys.extend(component)
#                 else:
#                     break

#         if len(testset_decoys) < self.num_decoy_per_sample:
#             logger.warning(f"pdb {name} facing similarity issue. No sufficient decoys for size_x_components! Please check the decoys and the tolerance.")
#             logger.warning(f"Trying to use size_over_x_components.")
#             for component in size_over_x_components:
#                 if len(component) < 2*self.num_decoy_per_sample:
#                     if len(validset_decoys) < self.num_decoy_per_sample:
#                         validset_decoys.extend(component)
#                     elif len(testset_decoys) < self.num_decoy_per_sample:
#                         testset_decoys.extend(component)
#                     else:
#                         break


#         if len(testset_decoys) < self.num_decoy_per_sample:
#             logger.warning(f"pdb {name} still facing similarity issue after using some of the size_over_x_components.")
#             logger.warning(f"valid set length:{len(validset_decoys)}")
#             logger.warning(f"test set length:{len(testset_decoys)}")
#             logger.warning(f"Only take the decoys above for valid/test subset.")

#         return validset_decoys, testset_decoys
