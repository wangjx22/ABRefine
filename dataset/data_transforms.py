import itertools
import torch

from utils.constants import residue_constants as rc
import utils.constants.atom_constants as atom_constants
from utils.opt_utils import gram_schmidt
import scipy
from scipy.spatial.transform import Rotation as R



from utils.logger import Logger

logger = Logger.logger


NUM_RES = "num residues placeholder"
NUM_MSA_SEQ = "msa placeholder"


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    is_gly = torch.eq(aatype, rc.res_type12id["G"])
    ca_idx = atom_constants.atom2id["CA"]
    cb_idx = atom_constants.atom2id["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


# @curry1
def make_pseudo_beta(protein):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    (protein["pseudo_beta"], protein["pseudo_beta_mask"],) = pseudo_beta_fn(
        protein["aatype"],
        protein["all_atom_positions"],
        protein["all_atom_mask"],
    )
    return protein


# @curry1
def make_fixed_size(
    protein,
    shape_schema,
    msa_cluster_size,
    num_res=0,
):
    """Guess at the MSA and sequence dimension to make fixed size."""
    pad_size_map = {
        NUM_RES: num_res,
        NUM_MSA_SEQ: msa_cluster_size,
    }

    for k, v in protein.items():
        # Don't transfer this to the accelerator.
        if k == "extra_cluster_assignment":
            continue
        shape = list(v.shape)
        schema = shape_schema[k]
        msg = "Rank mismatch between shape and shape schema for"
        assert len(shape) == len(schema), f"{msg} {k}: {shape} vs {schema}"
        pad_size = [pad_size_map.get(s2, None) or s1 for (s1, s2) in zip(shape, schema)]

        padding = [(0, p - v.shape[i]) for i, p in enumerate(pad_size)]
        padding.reverse()
        padding = list(itertools.chain(*padding))
        if padding:
            protein[k] = torch.nn.functional.pad(v, padding)
            protein[k] = torch.reshape(protein[k], pad_size)

    return protein


# @curry1
def select_feat(protein, feature_list):
    return {k: v for k, v in protein.items() if k in feature_list}


def make_atom37_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    protein_aatype = protein["aatype"].to(torch.long)

    # create the corresponding mask
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=protein["aatype"].device
    )
    for restype, res1 in enumerate(rc.restypes_1):
        res3 = rc.restype_1to3[res1]
        atom_names = rc.residue_atoms[res3]
        for atom_name in atom_names:
            atom_id = atom_constants.atom2id[atom_name]
            restype_atom37_mask[restype, atom_id] = 1

    # [N_res, 37]
    protein["atom37_atom_exists"] = restype_atom37_mask[protein_aatype]

    return protein


def atom37_to_frames(protein, eps=1e-12):
    """
    protein dict
    """
    # [N_res, 37, 3]
    all_atom_positions = protein["all_atom_positions"]
    # [N_res, 37]
    all_atom_mask = protein["all_atom_mask"]

    # backbone only
    bb_atom_name_list = ["C", "CA", "N"]
    bb_atom_id_list = [
        atom_constants.atom2id[atom_name] for atom_name in bb_atom_name_list
    ]

    ids = bb_atom_id_list
    base_atom_pos = all_atom_positions[:, ids, :]

    gt_R, gt_T, _ = gram_schmidt(
        p_neg_x_axis=base_atom_pos[..., 0, :],
        origin=base_atom_pos[..., 1, :],
        p_xy_plane=base_atom_pos[..., 2, :],
        eps=eps,
        enable_warning=True,
    )

    gt_atoms_exist = all_atom_mask[:, ids]
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0]

    protein["backbone_R"] = gt_R
    protein["backbone_T"] = gt_T
    protein["backbone_rigid_mask"] = gt_exists

    return protein


# @curry1
def random_crop_to_size(
    protein,
    crop_size,
    shape_schema,
):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    seq_length = protein["all_atom_positions"].shape[0]
    if crop_size:
        num_res_crop_size = min(int(seq_length), crop_size)
    else:
        num_res_crop_size = int(seq_length)

    def _randint(lower, upper):
        return int(
            torch.randint(
                lower,
                upper + 1,
                (1,),
                device=protein["all_atom_positions"].device,
            )[0]
        )

    n = seq_length - num_res_crop_size
    right_anchor = n

    num_res_crop_start = _randint(0, right_anchor)

    for k, v in protein.items():
        if k not in shape_schema or (
            "template" not in k and NUM_RES not in shape_schema[k]
        ):
            continue

        slices = []
        for i, (dim_size, dim) in enumerate(zip(shape_schema[k], v.shape)):
            is_num_res = dim_size == NUM_RES
            crop_start = num_res_crop_start if is_num_res else 0
            crop_size = num_res_crop_size if is_num_res else dim
            slices.append(slice(crop_start, crop_start + crop_size))
        protein[k] = v[slices]

    protein["seq_length"] = protein["seq_length"].new_tensor(num_res_crop_size)

    return protein


def rot_trans_noise_adding(rot, tran, rot_mean, rot_std, trans_mean, trans_std):
    """
    Adding noise to the rotation and translation.
    rot: Rotation matrix, shape: [batch, 3, 3] or [3,3]
    tran: transition vector, shape: [batch, 3] or [3]
    """
    # When using this method in the dataset/getitem, only one sample is processing, so the batch dim is squeezed.

    rot_shape = rot.size()
    # tran_shape = tran.size()

    rotation = R.from_matrix(rot)
    quat = torch.Tensor(rotation.as_quat())
    rot_noise = torch.randn_like(quat) * rot_std + rot_mean
    quat = quat + rot_noise
    rotation = R.from_quat(quat.numpy())
    new_rot = torch.Tensor(rotation.as_matrix())
    rot = new_rot.view(rot_shape)

    tran_noise = torch.randn_like(tran) * trans_std + trans_mean
    tran = tran + tran_noise
    return rot, tran


def pos_cg2res(
    cg_pos, N_res, scatter_mask, scatter_index, scatter_weight, deterministic=False
):
    """
    Args:
        cg_pos:
            [b, N_cg, 37, 3], cg position
        N_res:
            int, number of residue
        scatter_mask:
            [b, N_cg*37, 3], scatter mask
        scatter_index:
            [b, N_cg*37, 3], scatter index
        scatter_weight:
            [b, N_cg*37, 3], scatter weight
        deterministic:
            bool, whether to use deterministic algorithm
    Return:
        atom_pos:
            [b, N_res, 37, 3]
    """
    batch_size = cg_pos.shape[0]
    N_cg = cg_pos.shape[1]
    atom_pos = torch.zeros(
        (batch_size, N_res * 37, 3), dtype=torch.float32, device=cg_pos.device
    )
    mask_cg_pos = cg_pos.reshape(batch_size, N_cg * 37, 3) * scatter_mask
    if deterministic:
        # set deterministic = True for reproducibility in dev mode only
        torch.use_deterministic_algorithms(False)
    atom_pos2 = torch.scatter_add(
        atom_pos, 1, scatter_index, mask_cg_pos * scatter_weight
    )
    if deterministic:
        torch.use_deterministic_algorithms(True)
    atom_pos2 = atom_pos2.reshape(batch_size, N_res, 37, 3)

    return atom_pos2
