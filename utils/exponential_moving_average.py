"""Code."""
from collections import OrderedDict
import copy
import torch
import torch.nn as nn
from functools import partial


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    """Run dict_map method."""
    # code.
    new_dict = {}
    for k, v in dic.items():
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
    elif isinstance(tree, bool):
        return tree
    elif isinstance(tree, str):
        return tree
    else:
        print(type(tree))
        raise ValueError("Not supported")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)

class ExponentialMovingAverage:
    """Define Class ExponentialMovingAverage."""

    """
    Maintains moving averages of parameters with exponential decay

    At each step, the stored copy `copy` of each parameter `param` is
    updated as follows:

        `copy = decay * copy + (1 - decay) * param`

    where `decay` is an attribute of the ExponentialMovingAverage object.
    """

    def __init__(self, model: nn.Module, decay: float):
        """Run __init__ method."""
        # code.
        """
        Args:
            model:
                A torch.nn.Module whose parameters are to be tracked
            decay:
                A value (usually close to 1.) by which updates are
                weighted as part of the above formula
        """
        super(ExponentialMovingAverage, self).__init__()

        clone_param = lambda t: t.clone().detach()
        self.params = tensor_tree_map(clone_param, model.state_dict())
        self.decay = decay
        self.device = next(model.parameters()).device

    def to(self, device):
        """Run to method."""
        # code.
        self.params = tensor_tree_map(lambda t: t.to(device), self.params)
        self.device = device

    def _update_state_dict_(self, update, state_dict):
        """Run _update_state_dict_ method."""
        # code.
        with torch.no_grad():
            for k, v in update.items():
                stored = state_dict[k]
                if not isinstance(v, torch.Tensor):
                    self._update_state_dict_(v, stored)
                else:
                    diff = stored - v
                    diff = (
                        diff.float()
                    )  # Added to AntiBERTy, the positional encoding data type in AntiBERTy is long, but the decay is float, and an error will be reported.
                    diff *= (
                        1 - self.decay
                    )  # If we set the positional encoding data to float, then using it as an index in AntiBERTy throws an error.
                    stored = (
                        stored.float()
                    )  # Added for AntiBERTy, currently, we do not train AntiBERT.
                    stored -= diff

    def update(self, model: torch.nn.Module) -> None:
        """Run update method."""
        # code.
        """
        Updates the stored parameters using the state dict of the provided
        module. The module should have the same structure as that used to
        initialize the ExponentialMovingAverage object.
        """
        self._update_state_dict_(model.state_dict(), self.params)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Run load_state_dict method."""
        # code.
        self.params = state_dict["params"]
        self.decay = state_dict["decay"]

    def state_dict(self) -> OrderedDict:
        """Run state_dict method."""
        # code.
        return OrderedDict(
            {
                "params": self.params,
                "decay": self.decay,
            }
        )
