"""Code."""
# -*- encoding: utf-8 -*-
"""
@File    :   model_io.py
@Time    :   2021/10/05 18:39:55
@Author  :   Ming Ding
@Contact :   dm18@mails.tsinghua.edu.cn
"""

# here put the import lib
import os
import random
import torch
import numpy as np
import argparse

import utils.mpu_utils as mpu
from utils.logger import print_rank_0, Logger

logger = Logger.logger


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    """Run get_checkpoint_name method."""
    # code.
    if release:
        d = "release"
    else:
        d = "{:d}".format(iteration)
    if zero:
        dp_rank = mpu.get_data_parallel_rank()
        d += "_zero_dp_rank_{}".format(dp_rank)
    return os.path.join(
        checkpoints_path,
        d,
        "mp_rank_{:02d}_model_states.pt".format(mpu.get_model_parallel_rank()),
    )


def get_checkpoint_tracker_filename(checkpoints_path, old_checkpoint=False):
    """Run get_checkpoint_tracker_filename method."""
    # code.
    return os.path.join(checkpoints_path, "latest")


def extract_model_specific_args_from_model(args, model):
    """Run extract_model_specific_args_from_model method."""
    # code.
    parser = argparse.ArgumentParser()

    if hasattr(model, "module"):
        model = model.module
    if isinstance(model, torch.nn.Module):
        for md in model.modules():  # search
            if hasattr(md, "add_model_specific_args"):
                try:
                    md.add_model_specific_args(parser)
                except argparse.ArgumentError as e:
                    logger.error(e)
    ret = {}
    for k in vars(parser.parse_args([])).keys():
        if hasattr(args, k):
            ret[k] = getattr(args, k)
    return ret


def save_checkpoint(iteration, model, optimizer, lr_scheduler, args, client_state=None):
    """Run save_checkpoint method."""
    # code.
    """Save a model checkpoint."""
    if args.deepspeed:
        dp_rank = mpu.get_data_parallel_rank()
        logger.info(
            f"[DP Rank {dp_rank} is saving model checkpoint with args: save_method={args.save_method}"
        )
        if args.save_method == "model-only":
            if dp_rank == 0:
                save_ds_checkpoint(iteration, model, lr_scheduler, args)
        elif args.save_method == "model-n-optim":
            model.save_checkpoint(
                args.save, str(iteration), client_state=client_state, save_latest=True
            )
        else:
            assert False, f"Cannot recognize args.save_method {args.save_method}"
    else:
        raise ValueError("training without deepspeed is not supported.")
    # Wait so everyone is done (necessary)
    torch.distributed.barrier()
    # And update the latest iteration
    if torch.distributed.get_rank() == 0:
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    torch.distributed.barrier()


def save_ds_checkpoint(iteration, model, lr_scheduler, args):
    """Run save_ds_checkpoint method."""
    # code.
    """Save a model checkpoint."""

    sd = {}
    sd["iteration"] = iteration
    if lr_scheduler is not None:
        sd["client_lr_scheduler"] = lr_scheduler.state_dict()
    # rng states.
    if not args.no_save_rng:
        sd["random_rng_state"] = random.getstate()
        sd["np_rng_state"] = np.random.get_state()
        sd["torch_rng_state"] = torch.get_rng_state()
        sd["cuda_rng_state"] = torch.cuda.get_rng_state()

    save_ds_checkpoint_no_optim(model, args.save, str(iteration), client_state=sd)


def save_ds_checkpoint_no_optim(
    model, save_dir, tag=None, client_state={}, save_latest=True
):
    """Run save_ds_checkpoint_no_optim method."""
    # code.
    # Ensure save_dir directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Ensure tag is a string
    tag = str(tag)

    # Real save via deepspeed
    model._create_checkpoint_file(save_dir, tag, False)
    model._save_checkpoint(save_dir, tag, client_state=client_state)

    # Save latest checkpoint tag
    if save_latest:
        with open(os.path.join(save_dir, "latest"), "w") as fd:
            fd.write(tag)

    return True


def get_checkpoint_iteration(load_path):
    """Run get_checkpoint_iteration method."""
    # code.
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        print_rank_0("could not find the metadata file {} ".format(tracker_filename))
        # raise ValueError(
        #     "could not find the metadata file {}, please check --load".format(
        #         tracker_filename
        #     )
        # )
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, "r") as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == "release"
            if not release:
                print_rank_0(
                    "ERROR: Invalid metadata file {}. Exiting".format(tracker_filename)
                )
                exit()
    assert iteration > 0 or release, "error parsing metadata file {}".format(
        tracker_filename
    )

    return iteration, release, True


def load_checkpoint(model, args, load_path=None, prefix=""):
    """Run load_checkpoint method."""
    # code.
    """Load a model checkpoint."""
    if load_path is None:
        load_path = args.load

    iteration, release, success = get_checkpoint_iteration(load_path)
    if not success:
        return 0

    checkpoint_name = get_checkpoint_name(load_path, iteration, release)
    if mpu.get_data_parallel_rank() == 0:
        logger.info(
            "global rank {} is loading checkpoint {}".format(
                torch.distributed.get_rank(), checkpoint_name
            )
        )
    sd = torch.load(checkpoint_name, map_location="cpu")
    new_sd = {"module": {}}
    for k in sd:
        if k != "module":
            new_sd[k] = sd[k]
    for k in sd["module"]:
        if k.startswith(prefix):
            new_sd["module"][k[len(prefix) :]] = sd["module"][k]
    sd = new_sd

    if hasattr(model, "module"):
        module = model.module
    else:  # inference without deepspeed
        module = model

    # only load module, other hyperparameters are just for recording.
    missing_keys, unexpected_keys = module.load_state_dict(sd["module"], strict=False)
    if len(unexpected_keys) > 0:
        print_rank_0(
            f"Will continue but found unexpected_keys! Check whether you are loading correct checkpoints: {unexpected_keys}."
        )

    if len(getattr(args, "lora_types", [])) > 0:
        _mk = missing_keys
        missing_keys = []
        for k in _mk:
            if k.rsplit(".", 1)[-1] in ["weight_a", "weight_b"]:
                print_rank_0(
                    f"WARNING!!! {k} is missing from the checkpoint, be sure it is how you expected!"
                )
            else:
                missing_keys.append(k)

    if len(missing_keys) > 0:
        if args.mode == "inference":
            raise ValueError(f"Missing keys for inference: {missing_keys}.")
        else:  # new params
            assert all(name.find("mixins") >= 0 for name in missing_keys), missing_keys
            assert args.mode == "finetune"
            # list all mixin names
            mixin_names = []
            for key_name in missing_keys:
                parts = key_name.split(".")
                mixin_name = parts[parts.index("mixins") + 1]
                if mixin_name not in mixin_names:
                    mixin_names.append(mixin_name)
            module.reinit(mixin_names)  # initialize mixins

    # Do not need this any more, because we create optimizer after load now.
    # if args.mode != 'inference' and args.deepspeed and args.fp16:
    #     model.optimizer.refresh_fp32_params() # restore fp32 weights from module

    # Iterations.
    if args.mode == "finetune":
        iteration = 0
    elif args.mode == "pretrain" and not args.no_load_rng:  # rng states.
        try:
            random.setstate(sd["random_rng_state"])
            np.random.set_state(sd["np_rng_state"])
            torch.set_rng_state(sd["torch_rng_state"])
            torch.cuda.set_rng_state(sd["cuda_rng_state"])
            mpu.get_cuda_rng_tracker().set_states(sd["rng_tracker_states"])
        except KeyError:
            print_rank_0(
                "Unable to load optimizer from checkpoint {}, exiting. "
                "Specify --no-load-rng or --finetune to prevent "
                "attempting to load the random "
                "state.".format(checkpoint_name)
            )
            exit()
    elif args.mode == "inference":
        module.eval()

    if mpu.get_data_parallel_rank() == 0:
        logger.info("  successfully loaded {}".format(checkpoint_name))
    del sd
    return iteration
