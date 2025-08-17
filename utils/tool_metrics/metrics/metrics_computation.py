"""Code."""
import os.path

import torch

from .metrics_dataclass import Metrics
from .rmsd import superimpose, cdr_global_superimpose
from .lddt import lddt
from .violation import clash_metrics

from utils.tool_metrics.protein import PDBFeature
from utils.tool_metrics.protein import residue_constants
from utils.tool_metrics.protein.feature import extract_seqs
import re
from utils.tool_metrics.utils import Logger
from Bio import Align
from typing import Tuple, Any, Sequence, Callable, Optional, List
from utils.tool_metrics.numbering import Numbering
aligner = Align.PairwiseAligner()
logger = Logger.logger


def find_max_continue_seq(nums):
    """Run  find_max_continue_seq method."""
    # code.
    candidate = []
    if not nums:
        return []
    current_length = 1
    start_index = 0
    first_continue_flag = 0
    for i in range(1, len(nums)):
        if nums[i] - nums[i - 1] == 1:
            current_length += 1
            first_continue_flag = 1
        else:
            end_index = start_index + current_length
            candidate.append((start_index, end_index))
            first_continue_flag = 0
            current_length = 1
            start_index = i
    candidate.append((start_index, start_index + current_length))
    out_idx = None
    max_l = 0
    for i, (s,e) in enumerate(candidate):
        cur_l =  e - s
        if cur_l > max_l:
            out_idx = i
            max_l = cur_l
    return candidate[out_idx]

def fill_up_index(gt_residue_index, gt_sequence, fill_char="X"):
    """Run  fill_up_index method."""
    # code.
    new_gt_residue_index = []
    new_gt_sequence = ""
    for i, (res_i, aa) in enumerate(zip(gt_residue_index, gt_sequence)):
        if i == 0:
            lasta_res_i = res_i
            new_gt_residue_index.append(res_i)
            new_gt_sequence += aa
            continue

        res_i_ = int(re.sub("[a-zA-Z]", "", str(res_i)))
        lasta_res_i_ = int(re.sub("[a-zA-Z]", "", str(lasta_res_i)))
        while res_i_ - lasta_res_i_ >1:
            new_gt_residue_index.append(lasta_res_i_ + 1)
            new_gt_sequence += fill_char
            lasta_res_i_ += 1

        new_gt_residue_index.append(res_i)
        new_gt_sequence += aa
        lasta_res_i = res_i
    return new_gt_residue_index, new_gt_sequence


def select_pdb_feature_by_chain(input_feat_dict, chain_ids):
    """Run  select_pdb_feature_by_chain method."""
    # code.
    out = {}
    for k, v in input_feat_dict.items():
        chain_index = input_feat_dict["chain_index"]
        mask = None
        for chain_id in chain_ids:
            _mask = (chain_index==int(chain_id))
            mask = _mask if mask is None else _mask * mask

        if isinstance(v, torch.Tensor):
            out[k] = v[mask]
        elif isinstance(v, List):
            out[k] = v[mask]
    return out


def get_rmsd_95(gt_coords_masked_ca, pred_coords_masked_ca, atom_mask_ca, superimposed_ca):
    """Run  select_pdb_feature_by_chain method."""
    # code.
    deltas = torch.square(gt_coords_masked_ca - superimposed_ca).sum(-1)
    threshold = torch.quantile(deltas, 0.95)
    indices = (deltas <= threshold)
    gt_coords_95, pred_coords_95, atom_mask_ca_95 = gt_coords_masked_ca[indices], pred_coords_masked_ca[indices], atom_mask_ca[indices]
    rot_ca_95, tran_ca_95, superimposed_95, rmsd_ca_95 = superimpose(
        gt_coords_95, pred_coords_95, atom_mask_ca_95
    )
    return gt_coords_95, pred_coords_95, atom_mask_ca_95, rot_ca_95, tran_ca_95, superimposed_95, rmsd_ca_95


class MetricsCompute:
    """Define Class  MetricsCompute:."""

    def __init__(
        self,
        inum: int,
        name: str,
        gt_feature: PDBFeature,
        pred_feature: PDBFeature,
        gt_path: str,
        pred_path: str,
        all_atom_rmsd: bool,
        cdr_metric: bool,
        compute_clash: bool,
        # timers,
        **kwargs
    ):
        """Run  __init__ method."""
        # code.
        self.inum = inum
        self.name = name
        self.gt_feature = gt_feature
        self.pred_feature = pred_feature
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.compute_all_atom_rmsd = all_atom_rmsd
        self.compute_cdr_metric = cdr_metric
        # self.timers = timers
        self.compute_clash = compute_clash

    @staticmethod
    def prepare_coords(gt_feature_dict, pred_feature_dict, return_ca=True):
        """Run  prepare_coords method."""
        # code.
        gt_coords = gt_feature_dict["all_atom_positions"]  # [N_res, 37, 3]
        pred_coords = pred_feature_dict["all_atom_positions"]

        all_atom_mask = gt_feature_dict["all_atom_mask"]  # [N_res, 37]

        gt_coords_masked = gt_coords * all_atom_mask[..., None]  # [N_res, 37, 3]
        pred_coords_masked = pred_coords * all_atom_mask[..., None]  # [N_res, 37, 3]

        ca_pos = residue_constants.atom_order["CA"]  # 1
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :]  # [N_res, 3]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :]  # [N_res, 3]
        atom_mask_ca = all_atom_mask[..., ca_pos]  # [N_res]
        if return_ca:
            return gt_coords_masked_ca, pred_coords_masked_ca, atom_mask_ca
        else:
            return gt_coords_masked, pred_coords_masked, all_atom_mask

    def cdr_metrics(self, gt_cdr_feature_dict, pred_cdr_feature_dict, rot_ca, tran_ca, mid="", suffix=""):
        """Run  cdr_metrics method."""
        # code.
        results = {}
        for i in range(1, 4):
            (
                gt_coords_masked_ca,
                pred_coords_masked_ca,
                atom_mask_ca,
            ) = self.prepare_coords(
                gt_cdr_feature_dict.get(f"cdr{i}"),
                pred_cdr_feature_dict.get(f"cdr{i}"),
            )
            cdr_rmsd = cdr_global_superimpose(
                gt_coords_masked_ca,
                pred_coords_masked_ca,
                atom_mask_ca,
                rot_ca,
                tran_ca,
            )
            results.update({f"CDR{mid}{i}_RMSD_CA{suffix}": cdr_rmsd})
        chain_type = gt_cdr_feature_dict.get("chain_type")
        assert chain_type in ['h', 'k', 'l'], f"unexpected chain_type: {chain_type}"
        prefix = 'H' if chain_type == 'h' else 'L'
        if not mid:
            results.update({"chain_type": chain_type})
        return results, prefix

    def compute_metrics(self):
        """Run  compute_metrics method."""
        # code.
        # timers = self.timers
        # timers(f"prepare computation").start()
        results = {}
        results.update({"pdb": self.name, "index": self.inum})

        gt_feature_dict = self.gt_feature.get_feature(is_gt=True, get_cdr_metric=True)
        pred_feature_dict = self.pred_feature.get_feature(is_gt=False, get_cdr_metric=True)

        aatype = gt_feature_dict["aatype"]
        gt_protein_seq = "".join(
            [residue_constants.restypes_with_x[aatype[i]] for i in range(len(aatype))]
        )

        aatype = pred_feature_dict["aatype"]
        pred_protein_seq = "".join(
            [residue_constants.restypes_with_x[aatype[i]] for i in range(len(aatype))]
        )

        self.gt_feature_dict = gt_feature_dict
        self.pred_feature_dict = pred_feature_dict

        # timers(f"prepare computation").stop()
        # logger.info(f"prepare computation takes {timers('prepare computation').elapsed_} sec")


        # timers(f"CA RMSD").start()
        # CA coords
        gt_coords_masked_ca, pred_coords_masked_ca, atom_mask_ca = self.prepare_coords(
            gt_feature_dict, pred_feature_dict
        )

        # CA RMSD
        rot_ca, tran_ca, superimposed_ca, rmsd_ca = superimpose(
            gt_coords_masked_ca, pred_coords_masked_ca, atom_mask_ca
        )

        results.update({"RMSD_CA": rmsd_ca})

        gt_coords_95, pred_coords_95, atom_mask_ca_95, rot_ca_95, tran_ca_95, superimposed_95, rmsd_ca_95 = get_rmsd_95(
            gt_coords_masked_ca, pred_coords_masked_ca, atom_mask_ca, superimposed_ca)
        for _ in range(4):
            gt_coords_95, pred_coords_95, atom_mask_ca_95, rot_ca_95, tran_ca_95, superimposed_95, rmsd_ca_95 = get_rmsd_95(
                gt_coords_95, pred_coords_95, atom_mask_ca_95, superimposed_95)
        results.update({"RMSD_CA_95": rmsd_ca_95})


        # timers(f"CA RMSD").stop()
        # logger.info(f"CA RMSD takes {timers('CA RMSD').elapsed_} sec")

        # timers(f"CDR RMSD").start()
        # CDR rmsd global alignment
        if self.compute_cdr_metric:
            gt_cdr_feature_dict = self.gt_feature.get_cdr_feature()
            if gt_cdr_feature_dict is not None:
                pred_cdr_feature_dict = self.pred_feature.get_cdr_feature()
                assert self.gt_feature.region_feat.shape[0] == self.pred_feature.region_feat.shape[0]

                cdr_results, prefix = self.cdr_metrics(gt_cdr_feature_dict, pred_cdr_feature_dict, rot_ca, tran_ca)
                cdr_results_95, prefix_95 = self.cdr_metrics(gt_cdr_feature_dict, pred_cdr_feature_dict, rot_ca_95,
                                                             tran_ca_95, suffix="_95")
                results.update(cdr_results)
                results.update(cdr_results_95)

                results.update(
                    {
                        f"len": torch.tensor(self.gt_feature_dict["aatype"].shape[0]),
                        f"len_CDR1": torch.tensor(gt_cdr_feature_dict["cdr1"]["aatype"].shape[0]),
                        f"len_CDR2": torch.tensor(gt_cdr_feature_dict["cdr2"]["aatype"].shape[0]),
                        f"len_CDR3": torch.tensor(gt_cdr_feature_dict["cdr3"]["aatype"].shape[0]),
                    }
                )
        # timers(f"CDR RMSD").stop()
        # logger.info(f"CDR RMSD takes {timers('CDR RMSD').elapsed_} sec")

        # all atom rmsd
        if self.compute_all_atom_rmsd:
            (
                gt_coords_masked_all,
                pred_coords_masked_all,
                atom_mask_all,
            ) = self.prepare_coords(gt_feature_dict, pred_feature_dict, return_ca=False)

            _, _, num_coords = pred_coords_masked_all.shape
            pred_coords_all_atoms = pred_coords_masked_all.view(-1, num_coords)
            gt_coords_all_atoms = gt_coords_masked_all.view(-1, num_coords)
            all_atom_mask_all_atoms = atom_mask_all.view(-1)
            _, _, _, rmsd_all_atom = superimpose(
                gt_coords_all_atoms, pred_coords_all_atoms, all_atom_mask_all_atoms
            )
            results.update({"RMSD_ALL_ATOMS": rmsd_all_atom})

        # clash
        if self.compute_clash:
            # timers("Clash").start()
            clashes = clash_metrics(pred_feature_dict)
            results.update(clashes)
            # timers("Clash").stop()
            # logger.info(f"Clash takes {timers('Clash').elapsed_} sec")
        return results

    def compute(self):
        """Run  compute method."""
        # code.
        # try:
        results = self.compute_metrics()
        # except Exception as e:
        #     metrics = None
        #     logger.error(f"{self.name} have an error....")
        #     gt_aa_seq = [residue_constants.aatype_to_str_sequence(self.gt_feature_dict["aatype"])]
        #     pred_aa_seq = [residue_constants.aatype_to_str_sequence(self.pred_feature_dict["aatype"])]
        #     logger.error(f"""
        #                 gt_aa_seq: {gt_aa_seq}
        #                 pred_aa_seq: {pred_aa_seq}
        #                 """)
        #     logger.error(e)
        return results
