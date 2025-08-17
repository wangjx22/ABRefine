"""Code."""
from prettytable import PrettyTable


def count_parameters(model):
    """Run count_parameters method."""
    # code.
    table = PrettyTable(["Modules", "Parameters"])
    num_trainable_params = 0
    num_frozen_parms = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if not parameter.requires_grad:
            num_frozen_parms += params
        else:
            table.add_row([name, params])
            num_trainable_params += params
    return table, num_frozen_parms, num_trainable_params
