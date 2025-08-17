"""Code."""
import os
import random

import numpy as np
import pandas as pd

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch

from np.protein import from_pdb_string
import np.residue_constants as rc
from dataset.feature.featurizer import process
from utils.data_encoding import encode_structure, encode_features, extract_topology
from dataset.screener import UpperboundScreener, ProbabilityScreener1d
from utils.logger import Logger
from utils.constants.atom_constants import *
from utils.constants.residue_constants import *

logger = Logger.logger


def get_residue_mask(A, B):
    """Run get_residue_mask method."""
    # code.
    n, m = A.size()
    B_expanded = B.unsqueeze(-1).expand(n, m)
    mask = A.bool()
    applied_samples = torch.where(mask, B_expanded, torch.zeros_like(A))
    C, _ = torch.max(applied_samples, dim=0)
    return C.to(torch.int)


def load_dataset(dataset_filepath):
    """Run load_dataset method."""
    # code.
    csvs = dataset_filepath.split(',')
    if len(csvs) == 1:
        return pd.read_csv(csvs[0])
    dfs = [pd.read_csv(csv) for csv in csvs]
    return pd.concat(dfs)


class Dataset(torch.utils.data.Dataset):
    """Define Class Dataset."""

    def __init__(
            self,
            dataset_filepath,
            config_data,
            features_flags=(True, False, False),
            mode='train',
    ):
        """Run __init__ method."""
        # code.
        super(Dataset, self).__init__()
        self.dataset_filepath = dataset_filepath
        self.config_data = config_data
        self.ftrs = [fn for fn, ff in zip(['qe', 'qr', 'qn'], features_flags) if ff]

        self.debug = config_data["debug"]
        self.mode = mode

        # read dataset from dataset filepath
        self.read_dataset_file(dataset_filepath)

        self.rmsd_threshold = self.config_data.get('rmsd_threshold', -1)
        self.cdr3_threshold = self.config_data.get('cdr3_threshold', -1)

        logger.info(f"{mode} set: {len(self.pdb)}")

        # self analyze
        if self.mode == "train" and self.config_data.get(f"self_analyze", False):
            logger.info(f"Analyzing training set...")
            self.analyze_config = self.config_data['self_analyze']
            self.self_analyze()
            logger.info(f"Distribution2d:{self.distribution2d}")
            logger.info(f"Distribution1d:{self.distribution1d}")

        if self.mode == 'train' and self.config_data.get(f"down_sampling", False):
            logger.info(f"Down sampling...")
            self.down_sampling(self.config_data['down_sampling'])
            logger.info(f"Down sampling finished.")

        if self.mode == 'train':
            self.index_by_pdbid()

        # update: screeners list
        self.screeners_config = config_data.get("screener", None)
        self.prepare_screeners()

    def index_by_pdbid(self):
        """Run index_by_pdbid method."""
        # code.
        self.data_indices_by_pdbid = {}
        total_num = len(self.down_sampling_indices) if "down_sampling_indices" in list(self.__dict__.keys()) else self.total_num

        for i in range(total_num):
            pdbid = self.pdb_id[i] if "down_sampling_indices" not in list(self.__dict__.keys()) else self.pdb_id[self.down_sampling_indices[i]]

            if pdbid not in self.data_indices_by_pdbid.keys():
                self.data_indices_by_pdbid[pdbid] = [i]
            else:
                self.data_indices_by_pdbid[pdbid].append(i)

    def down_sampling(self, down_sampling_configs):
        """Run down_sampling method."""
        # code.
        if "distribution1d" not in list(self.__dict__.keys()):
            raise RuntimeError(f"Down sampling need to run after self_analyze.")

        target = down_sampling_configs['target']
        self_probs = self.distribution1d[target]
        if target == "RMSD_CA":
            values = self.rmsd_ca
        elif target == "CDR3_RMSD_CA":
            values = self.cdr3_rmsd_ca
        else:
            raise RuntimeError(f"Down sampling target {target} is not supported yet.")

        bins = down_sampling_configs['bin']
        for item in self.analyze_config:
            if item['target'] == target:
                assert item['bin'] == bins, f"Down sampling bin should be {bins} but got {item['bin']}"

        goal_probs = down_sampling_configs.get('probs', None)
        size_ratio = down_sampling_configs['size_ratio']
        down_sampling_num = int(size_ratio * self.total_num)

        if goal_probs is None:
            logger.info(f"Down sampling according to the trainset distribution of {target}")
            logger.info(f"Down sampling number: {down_sampling_num}")

            origin_bin_counts = (self_probs * self.total_num).astype(int)
            goal_bin_counts = (self_probs * down_sampling_num).astype(int)

            all_index = np.arange(self.total_num)
            bin_indices = np.digitize(values, bins)

            bin_to_indices = {i: [] for i in range(1, len(bins))}
            for index, bin_index in zip(all_index, bin_indices):
                bin_to_indices[bin_index].append(index)

            goal_bin_indices = {}
            for i in range(len(bin_to_indices)):
                cur_bin_indices = bin_to_indices[i+1]

                if len(cur_bin_indices) > goal_bin_counts[i]:
                    random.shuffle(cur_bin_indices)
                    goal_bin_indices.update({i+1: cur_bin_indices[:goal_bin_counts[i]]})
                else:
                    tms = int(goal_bin_counts[i] / len(cur_bin_indices))
                    tmp_indices = cur_bin_indices * tms
                    random.shuffle(cur_bin_indices)
                    tmp_indices.extend(cur_bin_indices[:(goal_bin_counts[i] % len(cur_bin_indices))])
                    goal_bin_indices.update({i+1: tmp_indices})

            down_sampling_indices = []
            for bin_index, indices in goal_bin_indices.items():
                down_sampling_indices.extend(indices)

        else:
            logger.info(f"Down sampling according to the given testset {down_sampling_configs['source']} distribution of {target}")
            logger.info(f"Down sampling number: {down_sampling_num}")
            origin_bin_counts = (self_probs * self.total_num).astype(int)
            goal_bin_counts = (np.array(goal_probs) * down_sampling_num).astype(int)

            all_index = np.arange(self.total_num)
            bin_indices = np.digitize(values, bins)

            bin_to_indices = {i: [] for i in range(1, len(bins))}
            for index, bin_index in zip(all_index, bin_indices):
                bin_to_indices[bin_index].append(index)

            goal_bin_indices = {}
            for i in range(len(bin_to_indices)):
                cur_bin_indices = bin_to_indices[i + 1]

                if len(cur_bin_indices) > goal_bin_counts[i]:
                    random.shuffle(cur_bin_indices)
                    goal_bin_indices.update({i + 1: cur_bin_indices[:goal_bin_counts[i]]})
                else:
                    tms = int(goal_bin_counts[i] / len(cur_bin_indices))
                    tmp_indices = cur_bin_indices * tms
                    random.shuffle(cur_bin_indices)
                    tmp_indices.extend(cur_bin_indices[:(goal_bin_counts[i] % len(cur_bin_indices))])
                    goal_bin_indices.update({i + 1: tmp_indices})

            down_sampling_indices = []
            for bin_index, indices in goal_bin_indices.items():
                down_sampling_indices.extend(indices)

        self.down_sampling_indices = down_sampling_indices
        logger.info(f"Dataset size after down sampling:{len(self.down_sampling_indices)}")

    def prepare_screeners(self):
        """Run prepare_screeners method."""
        # code.
        if (self.screeners_config is not None) and (self.mode == 'train'):
            self.screeners = []
            for item in self.screeners_config:
                if item['class'] == "UpperboundScreener":
                    self.screeners.append(UpperboundScreener(item['upperbound'], item['name']))

                elif item['class'] == 'ProbabilityScreener1d':
                    if not self.config_data.get(f"self_analyze", False):
                        # no self_analyze, just screen samples according to the given probabilities and bins
                        bins = item['bin']
                        probs = item['probs']
                        assert len(bins) == len(probs) + 1
                        ps = {}
                        for i in range(len(probs)):
                            ps.update({bins[i+1]: probs[i]})
                        self.screeners.append(ProbabilityScreener1d(ps, item['name']))
                    else:
                        # self-analyze, the given item['probs'] is the distribution of the testset
                        # We will down sampling the trainset to make the distribution similar to the testset.
                        # Reject sampling
                        target = item['name']
                        bins = item['bin']
                        probs = item['probs']
                        self_probs = self.distribution1d[target]
                        assert len(probs) == len(self_probs)
                        new_probs = []
                        bin_counts = self_probs * self.total_num
                        bin_total = bin_counts / probs
                        down_sample_total_num = min(bin_total)
                        logger.info(f"Expected to sample {down_sample_total_num} items from the trainset for {target}.")
                        for i in range(len(probs)):
                            cur_bin_testset_prob = probs[i]
                            cur_bin_trainset_count = bin_counts[i]
                            cur_bin_sample_num = down_sample_total_num * cur_bin_testset_prob
                            new_probs.append(cur_bin_sample_num / cur_bin_trainset_count)

                        ps = {}
                        for i in range(len(new_probs)):
                            ps.update({bins[i + 1]: new_probs[i]})
                        self.screeners.append(ProbabilityScreener1d(ps, item['name']))

                else:
                    logger.warning(f"Given screener not implemented.")
        else:
            self.screeners = None

    def read_dataset_file(self, dataset_filepath):
        """Run read_dataset_file method."""
        # code.
        # Considering different input format for "dataset_filepath" config.
        if dataset_filepath.__class__ is not list:
            # Old format for dataset_filepath config.
            # dataset filepaths are provided as a string, with comma for splitting.
            # Split by comma, each part is a given dataset filepath.
            # As the valid/test path is still provided in this format, we should continue support this format.
            # But for training, this format is likely to be deprecated.
            logger.warning(f"Given dataset filepath is not a list. It will not be deprecated in the future. Please use a list of filepaths with other params.")
            if dataset_filepath.endswith('.txt'):
                # dataset files are given as a string of txts. Information is lack.
                # so it can not be used for training! But for inference and valid/test, it works.
                logger.warning(f"PDB files are provided as txt files. Please make sure that the pdbs are single chains, and have solved the cut-fv and missing residue issue!!!")
                with open(dataset_filepath, 'r') as f:
                    self.pdb = f.read().splitlines()
                if self.debug:
                    if self.mode == 'train':
                        self.pdb = self.pdb[:500]
                self.quality = [0] * len(self.pdb)
                self.chain = [None] * len(self.pdb)
                self.full_seq_AMRs = [None] * len(self.pdb)
                self.rmsd_ca = [np.nan] * len(self.pdb)
                self.rmsd_ca_95 = [np.nan] * len(self.pdb)
                self.cdr3_rmsd_ca = [np.nan] * len(self.pdb)
                self.cdr3_rmsd_ca_95 = [np.nan]* len(self.pdb)
                self.loss_weight = [1.0] * len(self.pdb)
                # self.pdb_id = ?     This cannot be used for training, so we don't need to provide pdb_id
                if self.mode == 'train':
                    raise RuntimeError(f".txt is not supported  now for training! Build csvs with other information.")
            else:
                # dataset files are given as a string of csvs. Information is given, but params, e.g. weights, are not given.
                # so defualt weights are used.
                # and it can be used for both training and inference.
                dataset_df = load_dataset(dataset_filepath)    # dataset_df is a merged df from multiple csvs.
                if self.debug:
                    if self.mode == 'train':
                        dataset_df = dataset_df[:500]

                gt_flag = False
                try:
                    self.pdb = list(dataset_df[self.config_data["pdb_fpath_col"]])
                except:
                    self.pdb = list(dataset_df['pdb_fpath'])
                    gt_flag = True


                try:
                    self.pdb_id = list(dataset_df['pdb_id']) if 'pdb_id' in dataset_df else list(dataset_df['id'])
                except:
                    import re
                    self.pdb_id = list(dataset_df['pdb']) if 'pdb' in dataset_df else [re.split("/", pdbfpath)[-1][:-4] for
                                                                                  pdbfpath in dataset_df['pdb_fpath']]
                    logger.warning(
                        f"Not a standard pdb_id column give. Please check whether the extracted pdb_id is correct (show 10 for sample): \n {self.pdb_id[:10]}")

                self.rmsd_ca_95 = list(dataset_df["rmsd_ca_95"]) if "rmsd_ca_95" in dataset_df else [0 if gt_flag else np.nan]*len(self.pdb)
                self.rmsd_ca = list(dataset_df["RMSD_CA"]) if "RMSD_CA" in dataset_df else [0 if gt_flag else np.nan]*len(self.pdb)
                self.cdr3_rmsd_ca_95 = list(dataset_df["CDR3_RMSD_CA_95"]) if "CDR3_RMSD_CA_95" in dataset_df else [0 if gt_flag else np.nan]*len(self.pdb)
                self.cdr3_rmsd_ca = list(dataset_df["CDR3_RMSD_CA"]) if "CDR3_RMSD_CA" in dataset_df else [0 if gt_flag else np.nan]*len(self.pdb)

                if "chain" in dataset_df:
                    self.chain = list(dataset_df["chain"])
                else:
                    logger.warning(f"No chain information is given. Please make sure that the pdbs are single chains.")
                    self.chain = [None] * len(self.pdb)

                if 'full_seq_AMR' in dataset_df:
                    self.full_seq_AMRs = list(dataset_df['full_seq_AMR'])
                else:
                    logger.warning(f"No full_seq_AMR information is given in. Please make sure that the cut-fv and MR issues are solved.")
                    self.full_seq_AMRs = [None] * len(self.pdb)

                self.loss_weight = list(dataset_df["loss_weight"]) if "loss_weight" in dataset_df else [1.0] * len(self.pdb)

        else:
            # New format for dataset_filepath config.
            # config information is provided as a list, each item corresponds to one dataset and its params.
            self.pdb = []
            self.chain = []
            self.rmsd_ca_95 = []
            self.rmsd_ca = []
            self.cdr3_rmsd_ca_95 = []
            self.cdr3_rmsd_ca = []
            self.chain = []
            self.loss_weight = []
            self.pdb_id = []
            self.full_seq_AMRs = []

            for item in dataset_filepath:
                # each item provide the path and other params of one dataset.
                filepath = item['path']
                weight = item['weight']
                loss_weight = item.get("loss_weight", 1.0)

                if filepath.endswith('.txt'):
                    # the dataset is provided as a txt file.
                    # each line in the txt file is a pdb file.
                    # As other metrics are not provided, this format can only be supported for inference.
                    # Training is invalid for this format!
                    logger.warning(
                        f"PDB files are provided as txt files. Please make sure that the pdbs are single chains, and have solved the cut-fv and missing residue issue!!!")
                    with open(filepath, 'r') as f:
                        pdb = f.read().splitlines()
                    if self.debug:
                        if self.mode == 'train':
                            pdb = pdb[:500]
                    quality = [0] * len(pdb)
                    chain = [None] * len(pdb)
                    full_seq_AMRs = [None] * len(pdb)
                    self.pdb.extend(pdb * weight)
                    self.quality.extend(quality * weight)
                    self.chain.extend(chain * weight)
                    self.full_seq_AMRs.extend(full_seq_AMRs * weight)
                    self.rmsd_ca_95.extend([np.nan] * len(pdb) * weight)
                    self.rmsd_ca.extend([np.nan] * len(pdb) * weight)
                    self.cdr3_rmsd_ca_95.extend([np.nan] * len(pdb) * weight)
                    self.cdr3_rmsd_ca.extend([np.nan] * len(pdb) * weight)
                    self.loss_weight.extend([1.0] * len(pdb) * weight)
                    # self.pdb_id = ?                This is not used for training, so we don't need to provide pdb_id.
                    if self.mode == 'train':
                        raise RuntimeError(f".txt is not supported now for training! Build csvs with other information.")

                else:
                    # typical format for training dataset.
                    # information are provided in csv, and other params are given in each item of the list.
                    dataset_df = load_dataset(filepath)
                    logger.info(f"dataset {filepath} contains {len(dataset_df)} samples.")
                    if self.debug:
                        if self.mode == 'train':
                            dataset_df = dataset_df[:500]

                    gt_flag = False
                    try:
                        pdb = list(dataset_df[self.config_data["pdb_fpath_col"]])
                    except:
                        pdb = list(dataset_df['pdb_fpath'])
                        gt_flag = True

                    try:
                        pdb_id = list(dataset_df['pdb_id']) if 'pdb_id' in dataset_df else list(dataset_df['pdb'])
                    except:
                        import re
                        pdb_id = list(dataset_df['id']) if 'id' in dataset_df else [re.split("/", pdbfpath)[-1][:-4] for pdbfpath in dataset_df['pdb_fpath']]
                        logger.warning(f"Not a standard pdb_id column give. Please check whether the extracted pdb_id is correct (show :10 for sample): \n {pdb_id[:10]}")

                    rmsd_ca_95 = list(dataset_df["rmsd_ca_95"]) if "rmsd_ca_95" in dataset_df else [0 if gt_flag else np.nan]*len(pdb)
                    rmsd_ca = list(dataset_df["RMSD_CA"]) if "RMSD_CA" in dataset_df else [0 if gt_flag else np.nan] * len(pdb)
                    cdr3_rmsd_ca_95 = list(
                        dataset_df["CDR3_RMSD_CA_95"]) if "CDR3_RMSD_CA_95" in dataset_df else [0 if gt_flag else np.nan]*len(pdb)
                    cdr3_rmsd_ca = list(dataset_df["CDR3_RMSD_CA"]) if "CDR3_RMSD_CA" in dataset_df else [0 if gt_flag else np.nan] * len(pdb)

                    if "chain" in dataset_df:
                        chain = list(dataset_df["chain"])
                    else:
                        logger.warning(f"No chain information is given. Please make sure that the pdb is single chain.")
                        chain = [None] * len(pdb)

                    if 'full_seq_AMR' in dataset_df:
                        full_seq_AMRs = list(dataset_df['full_seq_AMR'])
                    else:
                        logger.warning(
                            f"No full_seq_AMR information is given in. Please make sure that the cut-fv and MR issues are solved.")
                        full_seq_AMRs = [None] * len(pdb)

                    # loss weight (update)
                    lw = list(dataset_df["loss_weight"]) if "loss_weight" in dataset_df else [loss_weight] * len(pdb)

                    self.pdb.extend(pdb * weight)
                    self.rmsd_ca_95.extend(rmsd_ca_95 * weight)
                    self.rmsd_ca.extend(rmsd_ca * weight)
                    self.cdr3_rmsd_ca_95.extend(cdr3_rmsd_ca_95 * weight)
                    self.cdr3_rmsd_ca.extend(cdr3_rmsd_ca * weight)
                    self.chain.extend(chain * weight)
                    self.pdb_id.extend(pdb_id * weight)
                    self.loss_weight.extend(lw * weight)
                    self.full_seq_AMRs.extend(full_seq_AMRs * weight)


    def self_analyze(self):
        """Run self_analyze method."""
        # code.
        # distribution analyze
        d = len(self.analyze_config)
        self.distribution1d = {}
        self.distribution2d = {}
        if d == 2:
            ts = []
            bins = []
            values = []
            for i in range(d):
                t = self.analyze_config[i]['target']
                bin = self.analyze_config[i]['bin']
                if t.lower() == 'rmsd_ca':
                    value = np.array(self.rmsd_ca)
                elif t.lower() == 'cdr3_rmsd_ca':
                    value = np.array(self.cdr3_rmsd_ca)
                else:
                    raise NotImplementedError(f"{t} is not supported yet.")
                ts.append(t)
                bins.append(bin)
                values.append(value)

                hist, edges = np.histogram(value, bins=bin)
                prob = hist / len(value)
                self.distribution1d.update({t:prob})

            hist, x_edges, y_edges = np.histogram2d(values[0], values[1], bins=bins)
            prob = hist / len(values[0])
            self.distribution2d.update({f"{ts[0]}_{ts[1]}": prob})

        elif d == 1:
            t = self.analyze_config[0]['target']
            bin = self.analyze_config[0]['bin']
            if t.lower() == 'rmsd_ca':
                value = np.array(self.rmsd_ca)
            elif t.lower() == 'cdr3_rmsd_ca':
                value = np.array(self.cdr3_rmsd_ca)
            else:
                raise NotImplementedError(f"{t} is not supported yet.")

            hist, edges = np.histogram(value, bins=bin)
            prob = hist / len(value)
            self.distribution1d.update({t: prob})

        # other analyze
        self.total_num = len(self.pdb)

    def __len__(self):
        """Run __len__ method."""
        # code.
        if "down_sampling_indices" in list(self.__dict__.keys()):
            return len(self.down_sampling_indices)
        else:
            return len(self.pdb)

    def prepare_features(self, pdb, chain, full_seq_amr):
        """Run prepare_features method."""
        # code.
        with open(pdb, 'r') as f:
            pdb_str = f.read()
        po = from_pdb_string(pdb_str, chain_id=chain, ca_only=self.config_data["ca_only"], full_seq_AMR=full_seq_amr)
        atom_mask = po.atom_mask.reshape(-1).astype(bool)

        po2struc = {
            'atom_positions': 'xyz',
            'atom_names': 'name',
            'elements': 'element',
            'residue_names': 'resname',
            'resids': 'resid',
            'hetfields': 'het_flag',
            'icodes': 'icode'
        }
        structure = {}
        for k in [
            'atom_positions',
            'atom_names',
            'residue_names',
            'elements',
            'hetfields',
            'resids',
            'icodes',
        ]:
            item = eval(f"po.{k}")
            if k == 'atom_positions':
                item = item.reshape(-1, 3)
            else:
                item = item.reshape(-1, )
            item = item[atom_mask]
            structure[po2struc[k]] = item


        X = torch.from_numpy(structure['xyz'].astype(np.float32))
        atom_names = structure['name'].tolist()
        atom_numbers = torch.Tensor([atom2id[atname] for atname in atom_names])     # atom number of atoms in residues. max number is 37.

        elements = structure['element'].tolist()
        atomic_numbers = [element2atomic_numbers[element]for element in elements]
        atomic_numbers = torch.Tensor(atomic_numbers)

        res_names = structure['resname'].tolist()
        res1 = [restype_3to1[resname] for resname in res_names]
        resid = [res_type12id[res1[i]] for i in range(len(res1))]
        resid = torch.Tensor(resid)

        n_nodes = X.size(0)

        data = Data(pos=X, atom_numbers=atom_numbers, atomic_numbers=atomic_numbers, resid=resid, n_nodes=n_nodes)
        return data

    def __getitem__(self, k):
        """Run __getitem__ method."""
        # code.
        if k.__class__ == list:
            raise RuntimeError("This dataset does not support batch access.")

        if "down_sampling_indices" in list(self.__dict__.keys()):
            k = self.down_sampling_indices[k]

        # No matter whether down_sampling is used, k is the index number of entire dataset!
        # data information are fetched from the entire dataset by k!!!
        if self.mode == 'train':
            k0 = k
            pdbid0 = self.pdb_id[k0]
            rmsd0 = self.rmsd_ca[k0]
            cdr30 = self.cdr3_rmsd_ca[k0]

            try_times = 10
            passflag = False
            while try_times:
                if "down_sampling_indices" in list(self.__dict__.keys()):
                    k1 = self.down_sampling_indices[random.choice(self.data_indices_by_pdbid[pdbid0])]
                else:
                    k1 = random.choice(self.data_indices_by_pdbid[pdbid0])

                rmsd1 = self.rmsd_ca[k1]
                cdr31 = self.cdr3_rmsd_ca[k1]

                if abs(rmsd0 - rmsd1) < self.rmsd_threshold:
                    try_times -= 1
                elif abs(cdr30 - cdr31) < self.cdr3_threshold:
                    try_times -= 1
                else:
                    passflag = True
                    break
            if not passflag:
                logger.info(f"Cannot find sample with rmsd dev > {self.rmsd_threshold} and cdr3 dev > {self.cdr3_threshold}, skip.")
                return None

            pdbid1 = self.pdb_id[k1]

            assert pdbid1 == pdbid0, f"Two samples should be from the same PDB_id, but got {pdbid0} and {pdbid1}"

            data0 = self._get_one_sample(k0)
            data1 = self._get_one_sample(k1)

            if data0 is None or data1 is None:
                return None

            new_data = self.merge_data(data0, data1)
            return new_data
        else:
            data = self._get_one_sample(k)
            return [data]

    def merge_data(self,data0, data1):
        """Run merge_data method."""
        # code.
        return [data0, data1]

    def _get_one_sample(self, k):
        """Run _get_one_sample method."""
        # code.
        pdb = self.pdb[k]
        chain = self.chain[k]
        rmsd_ca_95 = self.rmsd_ca_95[k]
        rmsd_ca = self.rmsd_ca[k]
        cdr3_rmsd_ca_95 = self.cdr3_rmsd_ca_95[k]
        cdr3_rmsd_ca = self.cdr3_rmsd_ca[k]
        label = self.rmsd_ca_95[k] if self.config_data["label_col"] == "rmsd_ca_95" else self.rmsd_ca[k]  # old version
        loss_weight = self.loss_weight[k]
        full_seq_amr = self.full_seq_AMRs[k]

        # Screen
        # screen large rmsds
        # Compatible with previous version
        # Only "train" mode dataset need to be screened.
        if self.mode == 'train':
            rmsd_upbound = self.config_data.get("rmsd_upbound", None)
            if rmsd_upbound is not None:
                if rmsd_ca > rmsd_upbound:
                    logger.info(f"{pdb}, {chain}: {rmsd_ca} > {rmsd_upbound}")
                    return None

            cdr3_upbound = self.config_data.get("cdr3_upbound", None)
            if cdr3_upbound is not None:
                if cdr3_rmsd_ca > cdr3_upbound:
                    logger.info(f"{pdb}, {chain}: {cdr3_rmsd_ca} > {cdr3_upbound}")
                    return None

        if (self.screeners is not None) and (self.mode == 'train'):
            for screener in self.screeners:
                if screener.__class__ == UpperboundScreener:
                    if screener.target == 'RMSD_CA':
                        if not screener.screen(rmsd_ca):
                            logger.info(f"{pdb}, {chain}: {rmsd_ca} > {screener.upper_bound}, skip.")
                            return None

                    elif screener.target == 'CDR3_RMSD_CA':
                        if not screener.screen(cdr3_rmsd_ca):
                            logger.info(f"{pdb}, {chain}: {cdr3_rmsd_ca} > {screener.upper_bound}, skip.")
                            return None

                elif screener.__class__ == ProbabilityScreener1d:
                    if screener.target == 'RMSD_CA':
                        if not screener.screen(rmsd_ca):
                            logger.info(f"{pdb}, {chain}: {rmsd_ca} reject")
                            return None

                    elif screener.target == 'CDR3_RMSD_CA':
                        if not screener.screen(cdr3_rmsd_ca):
                            logger.info(f"{pdb}, {chain}: {cdr3_rmsd_ca} reject")
                            return None


        try:
            feats = self.prepare_features(pdb, chain, full_seq_amr)
        except Exception as e:
            logger.warning(f"Encountering error {type(e).__name__} when preparing features for {pdb}, {chain}")
            logger.warning(e)
            logger.warning(f"Ignore this item, return None.")
            return None

        # scale
        rmsd_scale_config = self.config_data.get("rmsd_scale", None)
        if rmsd_scale_config is None:
            quality = label
            rmsd_ca_scale = np.nan
            cdr3_rmsd_ca_scale = np.nan
            rmsd_ca_95_scale = np.nan
            cdr3_rmsd_ca_95_scale = np.nan

        elif rmsd_scale_config.__class__ is not list:
            quality = 1 / (1 + (label / self.config_data["scale_temp"]) ** self.config_data["scale_order"])
            rmsd_ca_scale = np.nan
            cdr3_rmsd_ca_scale = np.nan
            rmsd_ca_95_scale = np.nan
            cdr3_rmsd_ca_95_scale = np.nan

        else:
            rmsd_ca_scale = np.nan
            cdr3_rmsd_ca_scale = np.nan
            rmsd_ca_95_scale = np.nan
            cdr3_rmsd_ca_95_scale = np.nan

            for item in rmsd_scale_config:
                if item['target'] == "RMSD_CA":
                    l = rmsd_ca
                    if not np.isnan(l):
                        rmsd_ca_scale = 1 / (1 + (l / item["scale_temp"]) ** item["scale_order"])
                elif item['target'] == "CDR3_RMSD_CA":
                    l = cdr3_rmsd_ca
                    if not np.isnan(l):
                        cdr3_rmsd_ca_scale = 1 / (1 + (l / item["scale_temp"]) ** item["scale_order"])
                elif item['target'] == "RMSD_CA_95":
                    l = rmsd_ca_95
                    if not np.isnan(l):
                        rmsd_ca_95_scale = 1 / (1 + (l / item["scale_temp"]) ** item["scale_order"])
                elif item['target'] == "CDR3_RMSD_CA_95":
                    l = cdr3_rmsd_ca_95
                    if not np.isnan(l):
                        cdr3_rmsd_ca_95_scale = 1 / (1 + (l / item["scale_temp"]) ** item["scale_order"])
                else:
                    raise NotImplementedError(f"Scale of {item['target']} is not implemented yet.")

            quality = label

        feats.quality = quality
        feats.pdb = pdb
        feats.rmsd_ca_95 = rmsd_ca_95
        feats.rmsd_ca = rmsd_ca
        feats.cdr3_rmsd_ca_95 = cdr3_rmsd_ca_95
        feats.cdr3_rmsd_ca = cdr3_rmsd_ca
        feats.rmsd_ca_scale = rmsd_ca_scale
        feats.cdr3_rmsd_ca_scale = cdr3_rmsd_ca_scale
        feats.rmsd_ca_95_scale = rmsd_ca_95_scale
        feats.cdr3_rmsd_ca_95_scale = cdr3_rmsd_ca_95_scale
        feats.loss_weight = loss_weight
        # feats.update(
        #     {
        #         'quality': quality,
        #         'pdb': pdb,
        #         'rmsd_ca_95': rmsd_ca_95,
        #         'rmsd_ca': rmsd_ca,
        #         'cdr3_rmsd_ca_95': cdr3_rmsd_ca_95,
        #         'cdr3_rmsd_ca': cdr3_rmsd_ca,
        #         "rmsd_ca_scale": rmsd_ca_scale,
        #         "cdr3_rmsd_ca_scale": cdr3_rmsd_ca_scale,
        #         "rmsd_ca_95_scale": rmsd_ca_95_scale,
        #         "cdr3_rmsd_ca_95_scale": cdr3_rmsd_ca_95_scale,
        #         "loss_weight": loss_weight,
        #     }
        # )

        return feats

def data_list_collater(data_list):
    """Run data_list_collater method."""
    # code.
    data_list = [data for data in data_list if data is not None]
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