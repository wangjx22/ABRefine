"""Code."""
from functools import partial

import torch
from utils.tensor_utils import dict_multimap

from utils.logger import Logger

logger = Logger.logger


class OpenFoldBatchCollator:
    """Define Class OpenFoldBatchCollator."""

    def __call__(self, prots):
        """Run __call__ method."""
        # code.
        prots = list(filter(lambda x: x is not None, prots))  # remove failed data
        new_prots = []
        for i in prots:
            if torch.all(i["all_atom_positions"] == 0).item() == 1:
                continue
            else:
                new_prots.append(i)

        assert (
            len(prots) > 0
        ), "OpenFoldBatchCollator: Batch should contain at least one valid data"
        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, prots)
