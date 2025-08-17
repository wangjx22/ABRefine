"""Code."""
import warnings

import torch.distributed as dist
import argparse
import os
os.environ["WANDB_IGNORE_GIT"] = "1"
os.environ["WANDB_DISABLED"] = "false"  # 如果你仍想在 rank=0 使用 wandb
from utils import dist_utils
from utils.general import read_yaml_to_dict
from utils.logger import Logger

logger = Logger.logger
warnings.filterwarnings("ignore")

def arg_provider(parser):
    """Run arg_provider method."""
    # code.
    group = parser.add_argument_group("ScoreModel", "")

    def bool_type(bool_str: str):
        bool_str_lower = bool_str.lower()
        if bool_str_lower in ("false", "f", "no", "n", "0"):
            return False
        elif bool_str_lower in ("true", "t", "yes", "y", "1"):
            return True
        else:
            raise ValueError(f"Cannot interpret {bool_str} as bool")
    group.add_argument("--max-epochs", type=int, default=1000)


def main():
    """Run main method."""
    # code.
    try:
        # mp.set_start_method('spawn')  # or 'forkserver'
        if dist_utils.is_dist_avail_and_initialized() or True:

            from utils.arguments import initialize, set_random_seed
            args = initialize(arg_provider)        

            yaml_args = read_yaml_to_dict(args.yaml_path)
            # set_random_seed(yaml_args['config_runtime'].seed)
            logger.info(yaml_args)

            args.do_train = True
            args.do_valid = True
            args.do_test = True

            if args.version == 2:
                # from trainer import Trainer
                # trainer = Trainer(
                #     args, yaml_args["config_runtime"], yaml_args["config_model"], yaml_args["config_data"])
                # trainer.fit()
                print('ready to training!!!!!!!!')
                from dataset.sin_equiformerv2_dataset_check_moudle_loss import test_run_train
                
                test_run_train(args, yaml_args["config_runtime"], yaml_args["config_model"], yaml_args["config_data"])
            else:
                raise NotImplementedError("Given version of trainer is not available.")

    except Exception:
        import traceback
        logger.info(
            f"Rank={dist.get_rank()}, caught unexpected error: {traceback.format_exc()}"
        )


if __name__ == '__main__':
    main()
