"""Code."""
from functools import partial
import torch
import torch.nn as nn
from typing import List

from utils.logger import Logger

logger = Logger.logger


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    """Run permute_final_dims method."""
    # code.
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    """Run flatten_final_dims method."""
    # code.
    return t.reshape(t.shape[:-no_dims] + (-1,))


# def masked_mean(mask, value, dim, eps=1e-4):
#     mask = mask.expand(*value.shape)
#     return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


# def pts_to_distogram(pts, min_bin=2.3125, max_bin=21.6875, no_bins=64):
#     boundaries = torch.linspace(min_bin, max_bin, no_bins - 1, device=pts.device)
#     dists = torch.sqrt(torch.sum((pts.unsqueeze(-2) - pts.unsqueeze(-3)) ** 2, dim=-1))
#     return torch.bucketize(dists, boundaries)


def dict_multimap(fn, dicts):
    """Run dict_multimap method."""
    # code.
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if "label" == k or type(v) == str:
            new_dict[k] = all_v
            continue
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x, v_bins):
    """Run one_hot method."""
    # code.
    """
    Implements Algorithm 5.
    """
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    """Run batched_gather method."""
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


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    """Run dict_map method."""
    # code.
    new_dict = {}
    for k, v in dic.items():
        if "label" == k:
            new_dict[k] = v
            continue
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


def tree_map(fn, tree, leaf_type):
    """Run tree_map method."""
    # code.
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    elif type(tree) == str:
        return tree
    else:
        logger.info(type(tree))
        raise ValueError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def _fetch_dims(tree):
    """Run _fetch_dims method."""
    # code.
    shapes = []
    tree_type = type(tree)
    if tree_type is dict:
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif tree_type is list or tree_type is tuple:
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif tree_type is torch.Tensor:
        shapes.append(tree.shape)
    else:
        raise ValueError("Not supported")

    return shapes
