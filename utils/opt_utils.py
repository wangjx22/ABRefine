"""Code."""
import torch


from utils.logger import Logger

logger = Logger.logger


def masked_differentiable_rmsd(pos , ref, pos_mask):
    """Run masked_differentiable_rmsd method."""
    # code.
    """
    batch svd to compute R, T
    Args:
        pos:
            [b, M, 3], b: number of examples, M : number of atoms
        ref:
            [b, M, 3]
        pos_mask:
            [b, M]
    Returns:
        T:
            translation, [b, 3]
        R:
            rotation, [b, 3, 3]
    """
    if pos_mask is None:
        pos_mask = torch.ones(pos.shape[:2], device=pos.device)
    else:
        if pos_mask.shape[0] != pos.shape[0]:
            raise ValueError(
                "pos_mask should have same number of rows as number of input vectors."
            )
        if pos_mask.shape[1] != pos.shape[1]:
            raise ValueError(
                "pos_mask should have same number of cols as number of input vector dimensions."
            )
        if pos_mask.ndim != 2:
            raise ValueError("pos_mask should be 2 dimensional.")
    denom = torch.sum(pos_mask, dim=1, keepdim=True)
    denom[denom == 0] = 1.0
    # computer the central
    pos_mu = (
        torch.sum(pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
    )
    ref_mu = (
        torch.sum(ref * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
    )
    pos_c = pos - pos_mu
    ref_c = ref - ref_mu

    # Covariance matrix
    H = torch.einsum("bji,bjk->bik", ref_c, pos_mask[:, :, None] * pos_c)
    # logger.info([ref_c.shape, pos_c.shape, H.shape])

    U, S, Vh = torch.linalg.svd(H)
    # Decide whether we need to correct rotation matrix to ensure right-handed coord system
    locs = torch.linalg.det(U @ Vh) < 0
    S[locs, -1] = -S[locs, -1]
    U[locs, :, -1] = -U[locs, :, -1]
    # Rotation matrix
    R = torch.einsum("bji,bkj->bik", Vh, U)

    # Translation vector
    ref_mu_rot = torch.einsum("bij,bkj->bki", R, ref_mu) # 这个地方有问题，R加到ref中心节点上，直接导致偏移大，是否存在左乘右乘的问题
    T = pos_mu - ref_mu_rot
    T = T.squeeze(1)

    return T, R


def superimpose_single(reference, coords, differentiable=False, mask=None, eps=1e-6):
    """Run superimpose_single method."""
    # code.
    """

    Args:
        reference:
            [N, 3], reference matrix
        coords:
            [N, 3], coordinate matrix
        mask:
            [N], mask
    Returns:
        rot:
            [3, 3], rotation matrix
        tran:
            [3, 1], translation matrix
        rmsd:
            float, rmsd between reference and coords
        superimposed:
            [N, 3], numpy, transformed positions
    """
    reference = reference.unsqueeze(0)
    coords = coords.unsqueeze(0)
    if mask is not None:
        mask = mask.unsqueeze(0)

    if differentiable:
        tran, rot = masked_differentiable_rmsd(reference, coords, mask)
        superimposed_rot = torch.einsum("bij, bmj -> bmi", rot, coords)
        superimposed = superimposed_rot + tran
    else:
        """
        when enable differentiable rotation and translation, the training process is untable, the performance is worse.
        """
        d_mask = None if mask is None else mask.detach()
        tran, rot = masked_differentiable_rmsd(
            reference.detach(), coords.detach(), d_mask
        )
        superimposed_rot = torch.einsum("bij, bmj -> bmi", rot, coords)
        superimposed = superimposed_rot + tran

        

    
    # logger.info([rot.shape, coords.shape, tran.shape])
    # superimposed_rot = torch.einsum("bij, bmj -> bmi", rot, coords)
    # superimposed = superimposed_rot + tran
    # logger.info([rot.shape, coords.shape, tran.shape, superimposed.shape])
    if mask is not None:
        rmsd = torch.sqrt(
            torch.sum(torch.sum((superimposed - reference) ** 2, dim=-1) * mask)
            / (torch.sum(mask) + eps)
        )
    else:
        rmsd = torch.sqrt(
            torch.mean(torch.sum((superimposed - reference) ** 2, dim=-1))
        )

    rot = rot.squeeze(0)
    tran = tran.squeeze(0)

    return rot, tran, rmsd, superimposed


def batch_superimpose_single(
    reference, coords, differentiable=False, mask=None, eps=1e-6
):
    """Run batch_superimpose_single method."""
    # code.
    """

    Args:
        reference:
            [b, N, 3], reference matrix
        coords:
            [b, N, 3], coordinate matrix
        mask:
            [b, N], mask
    Returns:
        rot:
            [b, 3, 3], rotation matrix
        tran:
            [b, 3, 1], translation matrix
        rmsd:
            [b], rmsd between reference and coords
        superimposed:
            [b, N, 3], numpy, transformed positions
    """

    if differentiable:
        tran, rot = masked_differentiable_rmsd(reference, coords, mask)
    else:
        """
        when enable differentiable rotation and translation, the training process is untable, the performance is worse.
        """
        d_mask = None if mask is None else mask.detach()
        tran, rot = masked_differentiable_rmsd(
            reference.detach(), coords.detach(), d_mask
        )

    superimposed = torch.einsum("bij, bmj -> bmi", rot, coords) + tran[:, None, :]
    # logger.info([rot.shape, coords.shape, tran.shape, superimposed.shape, mask.shape])
    if mask is not None:
        rmsd = torch.sqrt(
            torch.sum(torch.sum((superimposed - reference) ** 2, dim=-1) * mask, dim=-1)
            / (torch.sum(mask, dim=-1) + eps)
            + eps
        )
    else:
        rmsd = torch.sqrt(
            torch.mean(torch.sum((superimposed - reference) ** 2, dim=-1), dim=-1) + eps
        )

    return rot, tran, rmsd, superimposed


def gram_schmidt(
    p_neg_x_axis: torch.Tensor,
    origin: torch.Tensor,
    p_xy_plane: torch.Tensor,
    eps: float = 1e-12,
    reverse: bool = True,
    warning_eps: float = 1e-5,
    enable_warning: bool = False,
    mask: torch.Tensor = None,
):
    """Run gram_schmidt method."""
    # code.
    """
    Gram-Schmidt algorithm, a stable numerical algorithm

    Args:
        p_neg_x_axis: [*, 3] coordinates
        origin: [*, 3] coordinates used as frame origins
        p_xy_plane: [*, 3] coordinates
        eps: Small epsilon value
    Returns:
        Rotation:
            [*, 3, 3]
        Translation:
            [*, 3]

    Reference:
        https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
        https://fgiesen.wordpress.com/2013/06/02/modified-gram-schmidt-orthogonalization/
    """
    if reverse:
        ex = p_neg_x_axis - origin
    else:
        ex = origin - p_neg_x_axis

    ey = p_xy_plane - origin
    ex_normalized = ex / (ex.pow(2).sum(dim=-1, keepdim=True) + eps).sqrt()

    ey_normalized = (
        ey
        - torch.einsum("...d, ...d -> ...", ey, ex_normalized)[..., None]
        * ex_normalized
    )
    ey_normalized = (
        ey_normalized / (ey_normalized.pow(2).sum(dim=-1, keepdim=True) + eps).sqrt()
    )

    eznorm = torch.cross(ex_normalized, ey_normalized)
    R = torch.stack([ex_normalized, ey_normalized, eznorm], dim=-1)
    T = origin

    if enable_warning:
        error = R.det() - 1
        if mask is not None:
            error = error * mask
        maxerr = error.abs().max()

        warn = False
        if maxerr > warning_eps:
            warn = True
            # logger.error(f"got non-orthogonal matrix: max-error={maxerr}")
        return R, T, warn
    else:
        return R, T
