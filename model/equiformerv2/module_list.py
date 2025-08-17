"""Code."""
import torch


class ModuleListInfo(torch.nn.ModuleList):
    """Define Class ModuleListInfo."""

    def __init__(self, info_str, modules=None):
        """Run __init__ method."""
        # code.
        super().__init__(modules)
        self.info_str = str(info_str)

    def __repr__(self):
        """Run __repr__ method."""
        # code.
        return self.info_str