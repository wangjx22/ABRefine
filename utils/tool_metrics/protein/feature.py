"""Code."""
import os
from typing import Mapping, List
import re
import numpy as np
import torch

from .protein import from_pdb_string, Protein
from . import residue_constants
from utils.tool_metrics.utils import Logger
from utils.tool_metrics.numbering import Numbering, get_regein_feature
from utils.tool_metrics.protein.residue_constants import restype_order_with_x_inv

from Bio.PDB import PDBParser

logger = Logger.logger
FeatureDict = Mapping[str, np.ndarray]


def get_seq_info(input_seq):
    """Run  get_seq_info method."""
    # code.
    i = 0
    cleard_seq = ""
    last_is_aa = True
    is_add = []
    for aa in input_seq:
        if aa in "[]":
            if aa == "[":
                cur_is_aa = False
        else:
            cleard_seq += aa
            if last_is_aa:
                is_add.append(0)
            else:
                is_add.append(1)
            cur_is_aa = True
        last_is_aa = cur_is_aa
        if aa == "X":
            is_add.pop()
            cleard_seq = cleard_seq[:-1]
    is_add = torch.tensor(is_add, dtype=torch.int)
    return cleard_seq, is_add


def get_clead_seq(seq, is_add):
    """Run  get_clead_seq method."""
    # code.
    if type(seq) in [list, str]:
        is_add = is_add.tolist()
        out = ""
        for i, aa in enumerate(seq):
            if not is_add[i]:
                out += aa
        return out
    else:
        return seq[is_add]


def get_number_info(protein_seq):
    """Run  get_number_info method."""
    # code.
    try:
        numbering = Numbering(protein_seq)
        _, region_feat = get_regein_feature(protein_seq, numbering)
    except Exception as e:
        numbering, region_feat = None, None
    return numbering, region_feat



class PDBFeature:
    """Define Class  PDBFeature:."""

    def __init__(self, pdb_str: str, is_temp_file=False, is_cut_ab_fv=False, chain_id=None, is_gt=True,
                full_seq_with_mark=None,
                ):
        """Run  __init__ method."""
        # code.
        self.is_temp_file = is_temp_file
        self.full_seq_with_mark = full_seq_with_mark
        self.protein_object = from_pdb_string(
            pdb_str, chain_id=chain_id, is_cut_ab_fv=is_cut_ab_fv, is_gt=is_gt,
            skip_x=True,
        )
        self.is_gt = is_gt
        protein_object = self.protein_object
        aatype = protein_object.aatype
        self.protein_seq = "".join(
            [residue_constants.restypes_with_x[aatype[i]] for i in range(len(aatype))]
        )


    def get_feature(self, is_gt=True, get_cdr_metric=True):
        """Run  get_feature method."""
        # code.
        is_gt = self.is_gt
        feature = self._get_feature(self.protein_object)
        if self.full_seq_with_mark is None:
            self.feature_dict = feature
            numbering, region_feat = get_number_info(self.protein_seq) if get_cdr_metric else (None, None)
            self.numbering = numbering
            self.region_feat = region_feat
            return feature
        else:
            feature = self._get_cleand_feature(feature, is_gt=is_gt, get_cdr_metric=get_cdr_metric)
            return feature


    def _get_cleand_feature(self, feature, is_gt=True, get_cdr_metric=True):
        """Run  _get_cleand_feature method."""
        # code.
        full_seq_with_mark = self.full_seq_with_mark
        protein_seq = self.protein_seq
        full_seq, is_add = get_seq_info(full_seq_with_mark)
        # print(f"fv_w_amr:{full_seq_with_mark}")
        # print(f"full_seq after get seq info:{full_seq}")
        # print(f"is add after get seq info:{is_add}")
        assert len(full_seq) == len(is_add)
        numbering, region_feat = get_number_info(full_seq) if get_cdr_metric else (None, None)
        self.numbering = numbering


        if is_gt:
            mask = (1 - is_add).bool()
            clearned_seq = get_clead_seq(full_seq, is_add)
            self.region_feat = get_clead_seq(region_feat, mask)  if region_feat is not None else None
            start_index = protein_seq.find(clearned_seq)
            # print(f"protein_seq:{protein_seq}")
            # print(f"clearned_seq:{clearned_seq}")
            # print(f"start_index:{start_index}")
            assert start_index != -1
            end_index = start_index + len(clearned_seq)

            index_tensor = torch.tensor([i for i in range(start_index, end_index)])
            out_feat = select_pdb_feature_by_index(feature, index_tensor)
        else:
            start_index = full_seq.find(protein_seq)
            end_index = start_index + len(protein_seq)

            full_seq = full_seq[start_index:end_index]
            is_add = is_add[start_index:end_index]
            region_feat = region_feat[start_index: end_index] if region_feat is not None else None
            mask = torch.logical_not(is_add)
            self.region_feat = region_feat[mask] if region_feat is not None else None
            out_feat = select_pdb_feature_by_mask(feature, mask)
        if self.region_feat is not None:
            assert self.region_feat.shape[0] == out_feat["aatype"].shape[0]
        self.feature_dict = out_feat
        return out_feat


    def _get_feature(self, protein_object):
        """Run  _get_feature method."""
        # code.
        chain_id = torch.Tensor(protein_object.chain_index + 1)
        pdb_feats = make_pdb_features(protein_object, "FEATURE")
        pdb_feats = {
            "all_atom_positions": torch.Tensor(
                pdb_feats["all_atom_positions"]
            ),
            "all_atom_mask": torch.Tensor(pdb_feats["all_atom_mask"]),
            "aatype": torch.Tensor(pdb_feats["aatype"]).argmax(-1),
            "residue_index": torch.Tensor(pdb_feats["residue_index"]),
            # "seq_length": torch.Tensor([pdb_feats["seq_length"][0]]),
            "chain_ids": pdb_feats["chain_ids"],
            "chain_index": torch.Tensor(pdb_feats["chain_index"]),
        }
        return pdb_feats

    def get_cdr_feature(self):
        """Run  get_cdr_feature method."""
        # code.
        region_feat = self.region_feat
        if region_feat is None:
            return None
        out_info = {
                    "chain_type": self.numbering.chain_type,
                    "accum_L": self.numbering.accum_L,
                    }
        for cdr_idx, cdr_mark in enumerate([2, 4, 6]):
            cdr_idx = cdr_idx + 1
            mask = region_feat == cdr_mark
            mask_string = "".join([str(i) for i in mask.astype(int).tolist()])
            match = re.search("1+", mask_string)
            start_index, end_index = match.start(), match.end()
            index_tensor = torch.tensor([i for i in range(start_index, end_index)])
            cdr_feat = select_pdb_feature_by_index(self.feature_dict, index_tensor)
            out_info[f"cdr{cdr_idx}"] = cdr_feat
        return out_info



def select_pdb_feature_by_index(input_feat_dict, index_tensor):
    """Run  select_pdb_feature_by_index method."""
    # code.
    index_list = index_tensor.tolist()
    index_set = set(index_list)
    out = {}
    for k, v in input_feat_dict.items():
        if isinstance(v, torch.Tensor) and len(v.shape)>=1:
            out[k] = v[index_tensor]
        elif isinstance(v, List):
            out[k] = [d for i, d in enumerate(v) if i in index_set]
    return out

def select_pdb_feature_by_mask(input_feat_dict, mask):
    """Run  select_pdb_feature_by_mask method."""
    # code.
    out = {}
    for k, v in input_feat_dict.items():
        if isinstance(v, torch.Tensor) and len(v.shape)>=1:
            out[k] = v[mask]
        elif isinstance(v, List):
            out[k] = [d for i, d in enumerate(v) if mask[i]]
    return out


def make_sequence_features(
    sequence: str, description: str, num_res: int, aa_type=None,
) -> FeatureDict:
    """Run  make_sequence_features method."""
    # code.
    """
    Construct a feature dict of sequence features.

    sequence-level features as shown in Table.1, page 8.

    aa_type:
        One-hot representation of the input amino acid sequence (20 amino acids + unknown).
    residue_index:
        The index into the original amino acid sequence, indexed from 0.
    target_feat in sec.1.2.9
        concatenation of aatype and between_segment_residues
    """
    features = dict()
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    onehot_aatype_ = torch.argmax(torch.tensor(features["aatype"]), dim=-1).tolist()
    assert aa_type.tolist() == onehot_aatype_
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=np.object_)
    # features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
    return features


def _aatype_to_str_sequence(aatype):
    """Run  _aatype_to_str_sequence method."""
    # code.
    return "".join(
        [residue_constants.restypes_with_x[aatype[i]] for i in range(len(aatype))]
    )


def make_pdb_features(
    protein_object: Protein,
    description: str,
) -> FeatureDict:
    """Run  make_pdb_features method."""
    # code.

    protein_object_keys = [name for  name in dir(protein_object) if "__" not in name]
    pdb_feats = {key: getattr(protein_object, key) for key in protein_object_keys}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
            aa_type=aatype,
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)
    pdb_feats["chain_ids"] = protein_object.chain_ids

    return pdb_feats


def extract_seqs(asym_id, aatype):
    """Run  extract_seqs method."""
    # code.
    cur_id = -1
    seqs = []
    accum_lens = [0]
    for i in range(asym_id.shape[-1]):
        if asym_id[0, i] != cur_id:
            seqs.append("")
            cur_id = asym_id[0, i]
            if len(seqs) > 1:
                accum_lens.append(accum_lens[-1] + len(seqs[-2]))
        seqs[-1] += residue_constants.restype_order_with_x_inv[aatype[0, i].item()]
    accum_lens.append(accum_lens[-1] + len(seqs[-1]))
    return seqs, accum_lens


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    """Run  batched_gather method."""
    # code.
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def make_atom14_positions(protein):
    """Run  make_atom14_positions method."""
    # code.
    """Constructs denser atom positions (14 dimensions instead of 37)."""
    residx_atom14_mask = protein["atom14_atom_exists"]
    residx_atom14_to_atom37 = protein["residx_atom14_to_atom37"]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * batched_gather(
        protein["all_atom_mask"],
        residx_atom14_to_atom37,
        dim=-1,
        no_batch_dims=len(protein["all_atom_mask"].shape[:-1]),
    )

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[..., None] * (
        batched_gather(
            protein["all_atom_positions"],
            residx_atom14_to_atom37,
            dim=-2,
            no_batch_dims=len(protein["all_atom_positions"].shape[:-2]),
        )
    )

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["atom14_gt_exists"] = residx_atom14_gt_mask
    protein["atom14_gt_positions"] = residx_atom14_gt_positions

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [
        residue_constants.restype_1to3[res] for res in residue_constants.restypes
    ]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(
            14,
            dtype=protein["all_atom_mask"].dtype,
            device=protein["all_atom_mask"].device,
        )
        for res in restype_3
    }
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        correspondences = torch.arange(14, device=protein["all_atom_mask"].device)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(source_atom_swap)
            target_index = residue_constants.restype_name_to_atom14_names[
                resname
            ].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = protein["all_atom_mask"].new_zeros((14, 14))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix
    renaming_matrices = torch.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[protein["aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum(
        "...rac,...rab->...rbc", residx_atom14_gt_positions, renaming_transform
    )
    protein["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = torch.einsum(
        "...ra,...rab->...rb", residx_atom14_gt_mask, renaming_transform
    )
    protein["atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = protein["all_atom_mask"].new_zeros((21, 14))
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = residue_constants.restype_order[
                residue_constants.restype_3to1[resname]
            ]
            atom_idx1 = residue_constants.restype_name_to_atom14_names[resname].index(
                atom_name1
            )
            atom_idx2 = residue_constants.restype_name_to_atom14_names[resname].index(
                atom_name2
            )
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    protein["atom14_atom_is_ambiguous"] = restype_atom14_is_ambiguous[protein["aatype"]]

    return protein


def make_atom14_masks(protein):
    """Run  make_atom14_masks method."""
    # code.
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]
        ]
        restype_atom14_to_atom37.append(
            [(residue_constants.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in residue_constants.atom_types
            ]
        )

        restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein["aatype"].to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=protein["aatype"].device
    )
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask

    return protein


def extract_seqs(asym_id, aatype):
    """Run  extract_seqs method."""
    # code.
    cur_id = -1
    seqs = []
    accum_lens = [0]
    for i in range(asym_id.shape[-1]):
        if asym_id[0, i] != cur_id:
            seqs.append("")
            cur_id = asym_id[0, i]
            if len(seqs) > 1:
                accum_lens.append(accum_lens[-1] + len(seqs[-2]))
        seqs[-1] += restype_order_with_x_inv[aatype[0, i].item()]
    accum_lens.append(accum_lens[-1] + len(seqs[-1]))
    return seqs, accum_lens
