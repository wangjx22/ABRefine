"""Code."""
from typing import List
import statistics
from dataclasses import dataclass, fields, InitVar

import pandas as pd
import torch


@dataclass
class Metrics:
    """Define Class  Metrics:."""

    def __init__(self, results_dict):
        """Run  __init__ method."""
        # code.
        for k in results_dict:
            if isinstance(results_dict[k], torch.Tensor):
                setattr(
                    self, k, round(float(results_dict[k].item()), 4)
                )  # Convert to float
            else:
                setattr(
                    self, k, results_dict[k]
                )  # Convert to float

    def log(self, logger):
        """Run  log method."""
        # code.
        field_names = [field.name for field in fields(type(self))]
        field_values = [str(getattr(self, field.name)) for field in fields(type(self))]
        width_name = statistics.mean(len(name) for name in field_names)
        width_value = statistics.mean(len(value) for value in field_values)
        width = int(max(width_name, width_value))

        field_names_str = "\t".join([f"{name:{width}}" for name in field_names])
        field_values_str = "\t".join([f"{value:{width}}" for value in field_values])

        logger.info(field_names_str)  # logger names
        logger.info(field_values_str)  # logger the values


def convert_metrics_to_dataframe(data):
    """Run  convert_metrics_to_dataframe method."""
    # code.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    out_pd = pd.DataFrame([d.__dict__ for d in data])
    if "index" in out_pd.columns:
        out_pd = out_pd.sort_values("index")
    return out_pd
