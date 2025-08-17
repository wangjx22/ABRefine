"""Code."""
"""Protein data type."""
import dataclasses
import io
import numpy as np
import torch
from Bio.PDB import PDBParser
from copy import deepcopy
import time
import random

from utils.antibody_utils import Numbering, ChothiaNumbering
import utils.constants.residue_constants as residue_constants
import utils.constants.atom_constants as atom_constants
import utils.constants.cg_constants as cg_constants
from utils.opt_utils import superimpose_single
from dataset.data_transforms import rot_trans_noise_adding, pos_cg2res

from utils.logger import Logger

logger = Logger.logger


@dataclasses.dataclass(frozen=True)
class Residue:
    """Define Class Residue."""

    res1: str

    # residue index extract from pdb
    res_idx: int

    # icode, e.g., '', A, B, C, D
    icode: str

    # original residue index
    org_res_idx: str

    # true refers to missing
    is_add: bool

    # [37, 3], atom coordinates
    pos: torch.FloatTensor

    # [37], 1 refers to exists, 0 refers to missing
    mask: torch.FloatTensor

    # [37], 1 refers to contain, 0 refers to not contain
    contain: torch.FloatTensor

    # [37], b_factors
    b_factors: torch.FloatTensor

    # cg features
    cg_feature: dict()


@dataclasses.dataclass(frozen=True)
class Chain:
    """Define Class Chain."""

    """representation for a single chain"""

    sequence: str

    pdb_id: str

    chain_id: str

    # bool, true refers to antibody
    is_ab: bool

    numbering: Numbering

    chothia_numbering: ChothiaNumbering

    # a list of residues
    residues: list


@dataclasses.dataclass(frozen=True)
class Protein:
    """Define Class Protein."""

    """representation for protein with multiple chains"""

    # pdb id
    pdb_id: str

    # the number of models
    num_models: int

    # number of chains in each model
    num_chains_list: list

    # protein resolution
    resolution: float

    # chain list
    chain: Chain


def get_chain_type(chain):
    """Run get_chain_type method."""
    # code.
    for i, res in enumerate(chain):
        if res.id[0] != " ":  # skip HETATM
            continue

        if res.resname.strip() in ["DA", "DC", "DG", "DT", "DI"]:
            return "DNA"
        elif res.resname.strip() in ["A", "C", "G", "U", "I"]:
            return "RNA"
        else:
            # Todo: update this definition.
            return "PROTEIN"


def compute_imagine_cg_pos(init_cg, atom_id_list, rot, tran):
    """Run compute_imagine_cg_pos method."""
    # code.
    """

    Args:
        init_cg:
            [9, 3], np.array, atom pos for given coarse graining
        atom_id_list:
            list, atom 37-id list for given coarse graining
        rot:
            [3, 3], rotation tensor
        tran:
            [3], translation tensor
    Return:
        all_imagine_atom_pos:
            [37, 3], imagine atom pos computed by coarse graining and rot/tran
    """
    atom_type_num = atom_constants.atom_type_num  # atom_type_num := 37
    all_imagine_atom_pos = torch.zeros((atom_type_num, 3), dtype=torch.float32)
    for k, atom_id in enumerate(atom_id_list):
        init_pos = torch.FloatTensor(init_cg[k])
        imagine_pos = torch.einsum("ij, j -> i", rot, init_pos) + tran
        all_imagine_atom_pos[atom_id] = imagine_pos

    return all_imagine_atom_pos


def extract_cg(
    res1,
    res_idx,
    position,
    mask,
    static_cg,
    first_k_atoms=3,
    org_res_idx=None,
    debug_info=None,
    warning=False,
    eps=1e-12,
    noise=None,
    region_name=None
):
    """Run extract_cg method."""
    # code.
    """

    Args:
        res1:
            residue name with short name
        position:
            atom position, [37, 3]
        mask:
            atom mask, [37], 1 refers to exist, 0 refers to nonexistent
        static_cg:
            static cg dict, [61, 9, 3]
        first_k_atoms:
            consider the first k atoms in each coarse graining
        org_res_idx:
            original residue index extracted from pdb
    """

    assert first_k_atoms == 3, "at least 3 three atoms, use larger number in the future"
    (
        cg_3_pos_list,
        cg_all_37pos_list,
        cg_all_imagine_37pos_list,
        cg_all_37mask_list,
        cg_gt_atom_mask_list,
        cg_id_list,
    ) = (list(), list(), list(), list(), list(), list())
    cg_mask_list = list()  # 1 refers to exists, 0 refers to missing
    cg_weight_list = list()
    cg_rot_list, cg_tran_list = (
        list(),
        list(),
    )  # ground truth rotation and translation
    feature = dict()
    rmsd_list = list()
    res3 = residue_constants.restype_1to3.get(res1)
    cgs = cg_constants.cg_dict.get(res3)
    res_id = residue_constants.res_type12id[res1]
    atom_type_num = atom_constants.atom_type_num  # atom_type_num := 37
    valid_atom_freq = [0] * atom_type_num

    for i in range(len(cgs)):
        cg = (res_id, i)
        cg_id = cg_constants.cg2id[cg]  # max(cg_id) = 61
        atom_id_list = cg_constants.cgid2atomidlist[cg_id]
        all_37atom_pos = torch.zeros((atom_type_num, 3), dtype=torch.float32)
        # actual atom mask in current cg
        all_37atom_mask = torch.zeros(atom_type_num, dtype=torch.int)
        # atom mask if all atoms in current cg exist
        all_atom_gt_mask = torch.zeros(atom_type_num, dtype=torch.int)
        k_atom_pos = [torch.zeros(3, dtype=torch.float32)] * first_k_atoms

        icg_atoms = cg_constants.cgid2atomidlist[cg_id]
        cg_pos = position[icg_atoms]
        mask_ = mask[icg_atoms]
        static_cg_pos = torch.FloatTensor(static_cg[cg_id][: len(icg_atoms), :])
        # todo, use batch_superimpose_single to accelerate.
        rot, tran, rmsd, superimpose = superimpose_single(
            cg_pos, static_cg_pos, mask=mask_
        )

        cg_mask = ~(mask_[:3] == 0).any() #and rmsd <= 1.0
        rot = rot if cg_mask else torch.eye(3, dtype=rot.dtype)
        tran = tran if cg_mask else torch.zeros(3, dtype=tran.dtype)
        # logger.info(tran)

        if cg_mask:
            all_imagine_atom_37pos = compute_imagine_cg_pos(
                static_cg[cg_id], atom_id_list, rot, tran
            )
        else:
            atom_type_num = atom_constants.atom_type_num  # atom_type_num := 37
            all_imagine_atom_37pos = torch.zeros(
                (atom_type_num, 3), dtype=torch.float32
            )
        for k, atom_id in enumerate(atom_id_list):
            # atom_name = atom_constants.atom_types[atom_id]
            pos = position[atom_id]
            all_37atom_pos[atom_id] = pos
            all_37atom_mask[atom_id] = mask[atom_id]
            all_atom_gt_mask[atom_id] = 1

            if k < first_k_atoms:
                k_atom_pos[k] = pos

            if cg_mask:
                # only add atom freq when cg_mask is true. used to calculate scatter weight
                # if the cg is valid, all the atoms in cg is valid, because you can infer all atoms' position
                # according to the computed rotation and translation
                valid_atom_freq[atom_id] += 1

        if cg_mask == 0 and not warning:
            # todo: explore in the future
            # logger.warning([res1, debug_info])
            warning = True
        cg_mask_list.append(cg_mask)  # 1 refers to exists, 0 refers to missing
        # cg_id = cg_constants.cg2id.get(cg)
        cg_id_list.append(cg_id)
        cg_all_37pos_list.append(all_37atom_pos)
        cg_all_37mask_list.append(all_37atom_mask)
        cg_gt_atom_mask_list.append(all_atom_gt_mask)

        # compute coarse-graining pos based on rotation and translation
        cg_all_imagine_37pos_list.append(all_imagine_atom_37pos)

        rmsd_list.append(rmsd)
        cg_rot_list.append(rot)
        cg_tran_list.append(tran)
    # loop for all cgs in this residue done.

    cg_weight = torch.FloatTensor(
        [1 / freq if freq > 0 else 0 for freq in valid_atom_freq]
    )
    for i in range(len(cgs)):
        if cg_mask_list[i]:
            cg_weight_list.append(cg_weight * cg_gt_atom_mask_list[i])
        else:
            zero = torch.zeros(atom_type_num, dtype=torch.float32)
            cg_weight_list.append(zero)

    feature["total_cg"] = len(cgs)
    feature["actual_cg"] = sum([v for v in cg_mask_list])
    # corner case: cg is existed, but some atoms in this cg is missing.
    feature["reset_atom_mask"] = (
        torch.FloatTensor([1 if freq > 0 else 0 for freq in valid_atom_freq]) * mask
    )
    feature["cg_mask"] = torch.FloatTensor(cg_mask_list)  # [N_cg], 1 refers to exist
    feature["cg_atom_weight"] = torch.stack(cg_weight_list)  # [N_cg, 37], atom weight
    # logger.info([res1, feature["cg_mask"], feature["cg_atom_weight"][:, 1], valid_atom_freq[1]])
    feature["cg_res_idx"] = torch.LongTensor(
        [res_idx] * len(cg_id_list)
    )  # [N_cg], each refers to residue index extracted from pdb
    feature["res_id"] = torch.LongTensor(
        [res_id] * len(cg_id_list)
    )  # [N_cg], each refers to residue id, each res_id in [0, 20]
    feature["cg_id"] = torch.LongTensor(cg_id_list)  # [N_cg], each cg_idx in [0, 60]
    feature["cg_all_37pos"] = torch.stack(
        cg_all_37pos_list
    )  # [N_cg, 37, 3], all atom positions order by atom_order
    feature["cg_all_imagine_37pos"] = torch.stack(
        cg_all_imagine_37pos_list
    )  # [N_cg, 37, 3], all imagine atom positions order by atom_order
    feature["cg_all_37mask"] = torch.stack(
        cg_all_37mask_list
    )  # [N_cg, 37], mask of all atom positions order by atom_order
    feature["cg_gt_atom_mask"] = torch.stack(
        cg_gt_atom_mask_list
    )  # [N_cg, 37], ground-truth mask of all atom positions order by atom_order
    feature["rmsd"] = torch.FloatTensor(rmsd_list)  # [N_cg]
    feature["cg_rot"] = torch.stack(cg_rot_list)  # [N_cg, 3, 3]
    feature["cg_tran"] = torch.stack(cg_tran_list)  # [N_cg, 3]

    return feature


def cg_decoys_generation(
        res1,
        res_idx,
        position,
        mask,
        static_cg,
        first_k_atoms=3,
        warning=False,
        noise=None,
        region_name=None
):
    """Run cg_decoys_generation method."""
    # code.
    """

    Args:
        res1:
            residue name with short name
        position:
            atom position, [37, 3]
        mask:
            atom mask, [37], 1 refers to exist, 0 refers to nonexistent
        static_cg:
            static cg dict, [61, 9, 3]
        first_k_atoms:
            consider the first k atoms in each coarse graining
        org_res_idx:
            original residue index extracted from pdb
    """
    add_noise_rate = noise.get("add_noise_rate", None)
    if add_noise_rate is not None:
        rnd = random.random()
        if rnd > add_noise_rate:
            return position

    decoy_all_cg_pos = []
    res3 = residue_constants.restype_1to3.get(res1)
    cgs = cg_constants.cg_dict.get(res3)
    atom_type_num = atom_constants.atom_type_num  # atom_type_num := 37
    res_id = residue_constants.res_type12id[res1]
    valid_atom_freq = [0] * atom_type_num
    for i in range(len(cgs)):
        cg = (res_id, i)
        cg_id = cg_constants.cg2id[cg]
        atom_id_list = cg_constants.cgid2atomidlist[cg_id]

        icg_atoms = cg_constants.cgid2atomidlist[cg_id]
        cg_origin_pos = position[icg_atoms]
        mask_ = mask[icg_atoms]
        static_cg_pos = torch.FloatTensor(static_cg[cg_id][: len(icg_atoms), :])
        rot, tran, rmsd, superimpose = superimpose_single(
            cg_origin_pos, static_cg_pos, mask=mask_
        )
        rot_mean = noise.rot_mean[region_name]
        rot_std = noise.rot_std[region_name]
        trans_mean = noise.trans_mean[region_name]
        trans_std = noise.trans_std[region_name]
        noise_rot, noise_tran = rot_trans_noise_adding(rot, tran, rot_mean, rot_std, trans_mean, trans_std)

        decoy_cg_atom_37pos = compute_imagine_cg_pos(
            static_cg[cg_id], atom_id_list, noise_rot, noise_tran
        )

        cg_mask = ~(mask_[:3] == 0).any() and rmsd <= 1.0
        if cg_mask:
            decoy_all_cg_pos.append(decoy_cg_atom_37pos)

        # if res3 == 'SER':
        #     print(f"cg_origin_pos:{cg_origin_pos}")
        #     print(f"decoy_cg_atom_37pos:{decoy_cg_atom_37pos}")

        for k, atom_id in enumerate(atom_id_list):
            if cg_mask:
                valid_atom_freq[atom_id]+=1


    cg_weight = torch.FloatTensor(
        [1 / freq if freq > 0 else 0 for freq in valid_atom_freq]
    )

    cg_weight = cg_weight.unsqueeze(-1)
    merged_decoy_all_cg_pos = torch.zeros(decoy_all_cg_pos[0].size())
    for x in decoy_all_cg_pos:
        merged_decoy_all_cg_pos += x * cg_weight

    return merged_decoy_all_cg_pos





def res_contain_mask(res3):
    """Run res_contain_mask method."""
    # code.
    atom_type_num = atom_constants.atom_type_num  # atom_type_num := 37
    contain = torch.zeros(
        (atom_type_num,)
    )  # 1 refers to contain, 0 refers to not contain
    for atom_name in residue_constants.residue_atoms[res3]:
        atom_idx = atom_constants.atom2id[atom_name]
        contain[atom_idx] = 1.0

    return contain


def format_residue_from_PDBResidue(
    res, res_idx, static_cg, icode=None, org_res_idx=None, debug_info=None, noise=None, region_name=None
):
    """Run format_residue_from_PDBResidue method."""
    # code.
    res3 = res.resname.strip()
    res1 = residue_constants.restype_3to1.get(res3, "X")
    if res1 == "X":
        logger.warning([res.resname.strip(), res1])
    atom_type_num = atom_constants.atom_type_num  # atom_type_num := 37
    pos = torch.zeros((atom_type_num, 3))
    mask = torch.zeros((atom_type_num,))  # 1 refers to exists, 0 refers to nonexistent
    contain = res_contain_mask(res3)
    b_factors = torch.zeros((atom_type_num,))
    expected_atoms = residue_constants.residue_atoms[res3].copy()
    expected_atom_cnt = len(expected_atoms)
    for atom in res:
        if atom.name not in atom_constants.atom_types:
            continue
        # todo, check the missing atoms in current residue
        atom_idx = atom_constants.atom2id[atom.name]
        pos[atom_idx] = torch.FloatTensor(atom.coord)
        mask[atom_idx] = 1.0
        b_factors[atom_idx] = atom.bfactor
        if atom.name in expected_atoms:
            expected_atoms.remove(atom.name)
        else:
            # todo: explore in the future
            # logger.warning(f"unknown atom.name={atom.name}")
            pass
    # position information is now in pos

    missing_atom_cnt = len(expected_atoms)
    if len(expected_atoms) > 0:
        debug_info = f"{debug_info}; missing {expected_atoms} in {res_idx}:{res1}"

    if noise is not None:
        # start_time = time.time()
        pos = cg_decoys_generation(
            res1,
            res_idx,
            pos,
            mask,
            static_cg,
            noise=noise,
            region_name=region_name
        )
        # print(f"time for adding noise: {time.time() - start_time}")

    cg_feature = extract_cg(
        res1,
        res_idx,
        pos, # position information
        mask,
        static_cg,
        org_res_idx=org_res_idx,
        debug_info=debug_info,
        noise=noise,
        region_name = region_name
    )
    # pos is the position information from PDB
    # if add noise to generate decoys, pos should also be modified.

    # update atom mask, when atom's corresponding cg is not existed, reset atom mask by 0.
    reset_mask = cg_feature["reset_atom_mask"]

    # if cg_feature["actual_cg"] == 0:
    #     logger.warn(debug_info)

    residue = Residue(
        res1=res1,
        res_idx=res_idx,
        icode=icode,
        org_res_idx=org_res_idx,
        is_add=False,
        pos=pos,       # Here, should be modified.
        mask=reset_mask,
        contain=contain,
        b_factors=b_factors,
        cg_feature=cg_feature,
    )

    return residue, expected_atom_cnt, missing_atom_cnt


def format_residue_from_resname(
    res1, res_idx, static_cg, icode=None, org_res_idx=None, debug_info=None
):
    """Run format_residue_from_resname method."""
    # code.
    atom_type_num = atom_constants.atom_type_num  # atom_type_num := 37
    pos = torch.zeros((atom_type_num, 3))
    mask = torch.zeros((atom_type_num,))  # 1 refers to exists, 0 refers to missing
    contain = torch.zeros(
        (atom_type_num,)
    )  # 1 refers to contain, 0 refers to not contain
    b_factors = torch.zeros((atom_type_num,))

    res3 = residue_constants.restype_1to3[res1]
    for atom_name in residue_constants.residue_atoms[res3]:
        atom_idx = atom_constants.atom2id[atom_name]
        contain[atom_idx] = 1.0

    cg_feature = extract_cg(
        res1,
        res_idx,
        pos,
        mask,
        static_cg,
        org_res_idx=org_res_idx,
        debug_info=debug_info,
    )

    residue = Residue(
        res1=res1,
        res_idx=res_idx,
        icode=icode,
        org_res_idx=org_res_idx,
        is_add=True,
        pos=pos,
        mask=mask,
        contain=contain,
        b_factors=b_factors,
        cg_feature=cg_feature,
    )
    return residue


def read_pdb(input_chain):
    """Run read_pdb method."""
    # code.
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

    return res_list, pdb_seq


def extract_chain(
    pdb_id,
    input_chain,
    static_cg,
    # is_cut_ab_fv=False,
    fv_w_amr=None,
    debug_info=None,
    # first_len=None,
    # skip_x=False,
    # numbering_times=1,  # todo, default=1
    warning_missing_res_cnt=5,
    warning_broken_atom_cnt=10,
    warning_missing_atom_ratio=0.3,
    warning_seq_len=90,
    warning_cg_cnt=90 * 2.5,
    warning_enable=False,
    eps=1e-8,
    # is_pred=False,  # used for offline evaluation
    mode=None,
    noise=None,
    MD=None
):
    """Run extract_chain method."""
    # code.
    seqs = list()
    res_list, pdb_seq = read_pdb(input_chain)
    # primary_seq, res_idx_list, is_add_list = profile
    # if len(primary_seq) != len(res_idx_list):
    #     logger.error(f"primary_seq={primary_seq}, res_idx_list={res_idx_list}")

    primary_fv_seq, is_add_list = get_seq_info(fv_w_amr)
    num = Numbering(primary_fv_seq.strip("X"), seq_is_fv=True)
    cnum = ChothiaNumbering(primary_fv_seq.strip("X"), seq_is_fv=True)
    if len(num.numbering_seq) == 0:
        num = Numbering(
            primary_fv_seq.strip("X"), bit_score_threshold=20, seq_is_fv=True
        )
        cnum = ChothiaNumbering(
            primary_fv_seq.strip("X"), bit_score_threshold=20, seq_is_fv=True
        )

    is_ab = num.is_antibody
    if warning_enable:
        assert is_ab, f"is_ab={is_ab}, primary_fv_seq={primary_fv_seq}"
    if is_ab:
        selected_seq = num.numbering_seq
    else:
        selected_seq = primary_fv_seq

    # update (ziqiao)
    # Get region mask for each residue
    if is_ab:
        fr1 = num.fr1
        fr2 = num.fr2
        fr3 = num.fr3
        fr4 = num.fr4
        cdr1 = num.cdr1
        cdr2 = num.cdr2
        cdr3 = num.cdr3
        total_region_len = len(fr1) + len(fr2) + len(fr3) + len(fr4) + len(cdr1) + len(cdr2) + len(cdr3)
        try:
            assert total_region_len == len(selected_seq)
        except Exception as e:
            raise e
        region_mask = torch.zeros(total_region_len)
        region_mask[0:len(fr1)] = 1
        region_mask[len(fr1):(len(fr1)+len(cdr1))] = 2
        region_mask[(len(fr1)+len(cdr1)):(len(fr1)+len(cdr1)+len(fr2))] = 3
        region_mask[(len(fr1)+len(cdr1)+len(fr2)):(len(fr1)+len(cdr1)+len(fr2)+len(cdr2))] = 4
        region_mask[(len(fr1)+len(cdr1)+len(fr2)+len(cdr2)):(len(fr1)+len(cdr1)+len(fr2)+len(cdr2)+len(fr3))] = 5
        region_mask[(len(fr1)+len(cdr1)+len(fr2)+len(cdr2)+len(fr3)):(len(fr1)+len(cdr1)+len(fr2)+len(cdr2)+len(fr3)+len(cdr3))] = 6
        region_mask[(len(fr1)+len(cdr1)+len(fr2)+len(cdr2)+len(fr3)+len(cdr3)):(len(fr1)+len(cdr1)+len(fr2)+len(cdr2)+len(fr3)+len(cdr3)+len(fr4))] = 7
        assert torch.sum(region_mask) == len(fr1) + 2*len(cdr1) + 3*len(fr2) + 4*len(cdr2) + 5*len(fr3) + 6*len(cdr3) + 7*len(fr4)
    # update over

    # if is_pred:
    #     primary_fv_seq = selected_seq
    be = primary_fv_seq.find(selected_seq)
    en = be + len(selected_seq)
    if be > 0:
        logger.warning(f"be={be}, debug_info={debug_info}")
    selected_is_add_list = is_add_list[be:en]
    selected_seq_no_amr = "".join(
        selected_seq[i] for i, is_add in enumerate(selected_is_add_list) if is_add == 0
    )
    reno_idx = 0  # renumber residue index based on its order and gap.
    residue_list = list()

    missing_res_cnt, broken_res_cnt, expected_atom_cnt, missing_atom_cnt, valid_cg = (
        0,
        0,
        0,
        0,
        0,
    )
    # logger.info(f"pdb_seq={pdb_seq}")
    # logger.info(f"selected_seq_no_amr={selected_seq_no_amr}")
    # logger.info(f"primary_fv_seq={primary_fv_seq}")
    if selected_seq_no_amr in pdb_seq:
        shift = pdb_seq.find(selected_seq_no_amr)
    else:
        logger.error(
            f"wrong sequence. selected_seq_no_amr={selected_seq_no_amr}, pdb_seq={pdb_seq}, debug_info={debug_info}"
        )
        raise False

    seq_len = len(selected_seq)   # with AMR length
    for i in range(seq_len):
        if selected_is_add_list[i] == 0:
            res = res_list[shift]
            shift += 1
        else:
            res = None

        if (
            res is None
            or residue_constants.restype_3to1.get(res.resname.strip(), "X") == "X"
        ):
            # todo: skip missing residue
            missing_res_cnt += 1
            res1 = selected_seq[i]
            residue = format_residue_from_resname(
                res1, reno_idx, static_cg, org_res_idx=-1, debug_info=debug_info
            )
        else:
            res1 = residue_constants.restype_3to1.get(res.resname.strip(), "X")
            if selected_seq[i] != res1:
                logger.error(
                    f"selected_seq[{i}] = {selected_seq[i]}, res1={res1}, idx={-1}"
                )
                raise False
            update_debug_info = f"idx={-1}, {debug_info}"

            # update(ziqiao)
            region_name_list = ['fr1','cdr1','fr2','cdr2','fr3','cdr3','fr4']
            region_name = region_name_list[int(region_mask[i]-1)]
            residue, exp_cnt, missing_cnt = format_residue_from_PDBResidue(
                res, reno_idx, static_cg, org_res_idx=-1, debug_info=update_debug_info, noise=noise, region_name=region_name
            )
            # update over

            if residue is None:
                missing_res_cnt += 1
                continue
            missing_atom_cnt += missing_cnt
            expected_atom_cnt += exp_cnt
            broken_res_cnt += 1 if missing_cnt > 0 else 0
            valid_cg += residue.cg_feature["actual_cg"]

        # if res1 == "X":
        #     logger.warning([i, "X"])
        seqs.append(res1)
        residue_list.append(residue)
        reno_idx += 1

    # if mode is not None and mode != "train":
    #     final_seq = "".join(seqs)
    #     logger.info(f"be={be}, en={en}, primary_fv_seq={primary_fv_seq}, "
    #                 f"selected_seq={selected_seq}, selected_seq_no_amr={selected_seq_no_amr}, final_seq={final_seq}")

    missing_ratio = missing_atom_cnt / (expected_atom_cnt + eps)
    if (
        missing_res_cnt > warning_missing_res_cnt
        or broken_res_cnt > warning_broken_atom_cnt
        or missing_ratio > warning_missing_atom_ratio
        or len(seqs) <= warning_seq_len
        or valid_cg < warning_cg_cnt
    ) and warning_enable:
        logger.warn(
            f"there are {missing_res_cnt}/{len(selected_seq)} missing residues, {broken_res_cnt}/{len(seqs)} broken residues, {missing_atom_cnt}/{expected_atom_cnt} missing atoms, {valid_cg} valid cgs. debug_info={debug_info}"
        )
    seqs = "".join(seqs)
    # logger.info(
    #     f"is_ab={is_ab}, selected_seq={selected_seq}, primary_seq={primary_seq}, final_seqs={seqs}"
    # )
    # assert selected_seq == seqs, f"selected_seq={selected_seq}, seqs={seqs}"

    chain = Chain(
        sequence=seqs,  # single chain sequence
        pdb_id=pdb_id,  # pdb id
        chain_id=input_chain.id,  # chain id
        is_ab=is_ab,  # bool, true refers to antibody
        numbering=num,  # numbering results
        chothia_numbering=cnum,  # chothia numbering
        residues=residue_list,  # a list of residues
    )

    return chain


def get_seq_info(input_seq):
    """Run  get_seq_info method."""
    # code.
    cleaned_seq = ""
    last_is_aa = True
    is_add_list = []
    for aa in input_seq:
        if aa in "[]":
            if aa == "[":
                cur_is_aa = False
        else:
            cleaned_seq += aa
            if last_is_aa:
                is_add_list.append(0)
            else:
                is_add_list.append(1)
            cur_is_aa = True
        last_is_aa = cur_is_aa
        if aa == "X":
            is_add_list.pop()
            cleaned_seq = cleaned_seq[:-1]

    return cleaned_seq, is_add_list


def extract_feature(
    static_cg,
    fv_w_amr: str,
    pdb_str: str,
    pdb_id: str,
    chain_id: str,
    resolution: float = None,
    # is_cut_ab_fv: bool = False,
    debug_info=None,
    # first_len=None,
    # skip_x=False,
    warning_enable=False,
    # is_pred=False,
    mode=None,
    noise=None,
    MD=None,
):
    """Run extract_feature method."""
    # code.
    """Takes a PDB string and constructs a Protein object.

    Args:
        static_cg: static coarse graining
        pdb_str: The contents of the pdb file
        chain_id: chain id

    Returns:
      A new `Protein` parsed from the pdb contents.
    """

    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    if resolution is None:
        resolution = structure.header["resolution"]
        if resolution is None:
            resolution = 0.0
    models = list(structure.get_models())
    num_models = len(models)
    num_chains_list = [len(list(model.get_chains())) for model in models]

    # only extract the first model, todo: further explore more models in the future
    model = models[0]
    pdb_chains = list(model.get_chains())

    pdb_chain = None
    for c in pdb_chains:
        if c.id.strip() == chain_id.strip():
            pdb_chain = c
            break

    assert pdb_chain is not None, f"cannot find chain {chain_id}."
    chain_type = get_chain_type(pdb_chain)
    assert chain_type == "PROTEIN", f"chain_type={chain_type}."

    chain_instance = extract_chain(
        pdb_id,
        pdb_chain,
        static_cg,
        # is_cut_ab_fv=False,
        fv_w_amr=fv_w_amr,
        debug_info=debug_info,
        # first_len=first_len,
        # skip_x=skip_x,
        warning_enable=warning_enable,
        # is_pred=is_pred,
        mode=mode,
        noise=noise,
        MD=MD
    )

    instance = Protein(
        pdb_id=pdb_id,
        num_models=num_models,
        num_chains_list=num_chains_list,
        resolution=resolution,
        chain=chain_instance,
    )

    all_extra = dict()

    return instance, all_extra


def to_pdb(res_id, seq_mask, res_37, chain_id="A", save="all", atoms=None, atom_masks=None) -> str:
    """Run to_pdb method."""
    # code.
    """Converts tensors to a PDB string.

    Args:
      res_id:
        [N_res], -1 refers to padding residue
      seq_mask:
        [N_res], 0 refers to padding residue
      res_37:
        [N_block, N_res, 37, 3]
      chain_id:
        chain id, default = A
    Returns:
      PDB string.
    """
    pdb_lines = []
    num_blocks = res_37.shape[0]
    if save == "all":
        block_id_list = range(num_blocks)
    else:
        # save last
        block_id_list = [-1]
    for b, block_id in enumerate(block_id_list):
        pdb_lines.append(f"MODEL     {b+1}")
        atom_index = 1
        b_factor = 0
        for i in range(res_id.shape[0]):
            # if i == 6:
            #     res_1 = residue_constants.res_id2type1[int(res_id[i])]
            #     res_3 = residue_constants.restype_1to3[res_1]
            #     print(res_1)
            #     print(res_3)
            #     print(atom_masks[i])
            #     atom_list = residue_constants.residue_atoms[res_3]
            #     print(atom_list)
            #     raise RuntimeError

            if int(seq_mask[i]) == 0:
                continue
            res_1 = residue_constants.res_id2type1[int(res_id[i])]
            res_3 = residue_constants.restype_1to3[res_1]
            if atoms:
                atom_list = atoms.split(",")
            else:
                atom_list = residue_constants.residue_atoms[res_3]
            atom_mask = atom_masks[i]
            for atom_name in atom_list:
                atom_name = atom_name.strip()
                atom_id = atom_constants.atom2id[atom_name]
                if atom_mask[atom_id] == 0:
                    continue

                record_type = "ATOM"
                name = atom_name if len(atom_name) == 4 else f" {atom_name}"
                alt_loc = ""
                insertion_code = ""
                occupancy = 1.00
                element = atom_name[0]  # Protein supports only C, N, O, S, this works.
                charge = ""
                atom_id = atom_constants.atom2id[atom_name]
                pos = res_37[block_id][i][atom_id]
                # PDB is a columnar format, every space matters here!
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                    f"{res_3:>3} {chain_id:>1}"
                    f"{i+1:>4}{insertion_code:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                pdb_lines.append(atom_line)
                atom_index += 1

        # Close the chain.
        chain_end = "TER"
        chain_termination_line = (
            f"{chain_end:<6}{atom_index-1:>5}      {res_3:>3} " f"{chain_id:>1}{i+1:>4}"
        )
        pdb_lines.append(chain_termination_line)
        pdb_lines.append("ENDMDL")

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)
