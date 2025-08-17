"""Code."""
import argparse

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from dataset.datasetv2 import collate_batch_data

from utils.general import read_yaml_to_dict
from model.Model import ScoreModel
from utils.logger import Logger

logger = Logger.logger


def get_args():
    """Run get_args method."""
    # code.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="/pfs_beijing/sunyiwu/structure_pretraining/interaction-learning/notebooks/va1set_1.csv",
        help="path of coarse graining",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/pfs_beijing/sunyiwu/structure_pretraining/interaction-learning-train/interaction-learning/ckpts/1_35000.ckpt",
        help="path of coarse graining",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="test_outputs.txt",
        help="path of coarse graining",
    )
    parser.add_argument(
        "--interface_cut",
        type=int,
        default=225,
        help="interface cutoff",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--yaml-path",
        type=str,
        default="",
        help="path of yaml file",
    )

    args = parser.parse_args()

    return args


def main(args):
    """Run main method."""
    # code.
    yaml_args = read_yaml_to_dict(args.yaml_path)
    try:
        device = torch.cuda.current_device()
        model = ScoreModel(yaml_args["config_model"])
        model.to(device)

        if args.ckpt_path.endswith(".pt"):
            state_dict = torch.load(args.ckpt_path)
            if "ema" in state_dict:
                model_dict = model.state_dict()
                state_dict = {
                    ".".join(k.split(".")[1:]): v
                    for k, v in state_dict["ema"]["params"].items()
                    if ".".join(k.split(".")[1:]) in model_dict
                }
        model.load_state_dict(state_dict)

        if yaml_args.config_data.dataset_type == "Datasetv2":
            from dataset.datasetv2 import Dataset
        elif yaml_args.config_data.dataset_type == 'Datasetv3':
            from dataset.datasetv3 import Dataset

        test_data = Dataset(
            args.input_path,
            yaml_args['config_data'],
            mode="test",
        )

        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            num_workers=args.batch_size - 1 if args.batch_size > 1 else 10,
            shuffle=False,
            collate_fn=collate_batch_data,
            pin_memory=True,
            prefetch_factor=(args.batch_size * 3) if args.batch_size > 1 else 10,
        )

        with torch.no_grad():
            model.eval()
            test_results, inputs = [], []
            out_weight = torch.Tensor(yaml_args['config_model']['out_weight']).to(device)
            for batch_data in tqdm(test_loader, total=len(test_loader)):
                # pdbs = batch_data[-1]
                # if batch_data[0] == None:
                #     z = [0]
                #     test_results += z
                # else:
                #     X, ids_topk, q, M, y, r_mask, target, fake = [data.to(device) for data in batch_data[:-1]]
                #     z = model(X, ids_topk, q, M, r_mask)
                #     test_results += z.detach().cpu().numpy().tolist()

                pdbs = batch_data['pdb']
                for k in batch_data:
                    if k != "pdb":
                        batch_data[k] = batch_data[k].to(device)
                z = model(batch_data)
                pred = torch.matmul(z, out_weight)

                # test_results += z.detach().cpu().numpy().tolist()
                test_results += pred.detach().cpu().numpy().tolist()

                inputs += pdbs
        with open(args.output_path, 'w') as f:
            for input_file, output in zip(inputs, test_results):
                f.write(f"{input_file},{output}\n")

    except Exception:
        import traceback
        logger.info(
            f"Rank={dist.get_rank()}, caught unexpected error: {traceback.format_exc()}"
        )


if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
