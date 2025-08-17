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

from Bio.SVDSuperimposer import SVDSuperimposer
import numpy as np
import torch


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

    return rot, tran, superimposed, rmsd


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
