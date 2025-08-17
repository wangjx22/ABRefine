"""Code."""
import argparse
import os
import torch
import deepspeed
import json
import random
import numpy as np
import utils.mpu_utils as mpu
from utils.deepspeed import version_ge_0_10 as deepspeed_version_ge_0_10

from utils.logger import Logger

logger = Logger.logger


def add_summary_writer_args(parser):
    """Run add_summary_writer_args method."""
    # code.
    group = parser.add_argument_group("summary_writer", "summary_writer configurations")
    group.add_argument(
        "--wandb-key",
        type=str,
        default="",
    )
    group.add_argument(
        "--wandb-name",
        type=str,
        default="",
    )
    group.add_argument(
        "--wandb", action="store_true", help="wether use wandb for summary writer"
    )
    group.add_argument(
        "--strict_eval", action="store_true", help="strict_eval"
    )
    group.add_argument(
        "--tensorboard",
        action="store_true",
        help="wether use tensorboard for summary writer",
    )
    return parser


def add_training_args(parser):
    """Run add_training_args method."""
    # code.
    """Training arguments."""

    group = parser.add_argument_group("train", "training configurations")

    # --------------- Core hyper-parameters ---------------
    group.add_argument(
        "--debug",
        action="store_true",
        help="whether to entry a debug mode.",
    )

    group.add_argument(
        "--override_some_deepspeed_config",
        type=str,
        default=None,
        help="which deepspeed config to be override",
    )
    group.add_argument(
        "--commit-id",
        type=str,
        default=None,
        help="commit id of this run",
    )
    group.add_argument(
        "--experiment-name",
        type=str,
        default="MyModel",
        help="The experiment name for summary and checkpoint."
        "Will load the previous name if mode==pretrain and with --load ",
    )
    group.add_argument(
        "--yaml-path",
        type=str,
        default="",
        help="path of yaml file",
    )
    group.add_argument(
        "--train-iters",
        type=int,
        default=1000000,
        help="total number of iterations to train over all training runs",
    )
    group.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="batch size on a single GPU. batch-size * world_size = total batch_size.",
    )
    group.add_argument("--lr", type=float, default=1.0e-4, help="initial learning rate")
    group.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=[
            "pretrain",  # from_scratch / load ckpt for continue pretraining.
            "finetune",  # finetuning, auto-warmup 100 iters, new exp name.
            "inference",  # don't train.
        ],
        help="what type of task to use, will influence auto-warmup, exp name, iteration",
    )
    group.add_argument("--seed", type=int, default=1234, help="random seed")
    group.add_argument(
        "--zero-stage",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="deepspeed ZeRO stage. 0 means no ZeRO.",
    )
    group.add_argument(
        "--decoys_exp",
        default=False,
        action="store_true",
        help="Only generate a set of decoys for analysis.",
    )
    group.add_argument(
        "--decoys_per_sample",
        type=int,
        default=1,
        help="number of decoys to generate for analysis",
    )
    group.add_argument(
        "--decoys_sample_num",
        type=int,
        default=1,
        help="number of samples from trainset to generate decoys for analysis",
    )
    group.add_argument(
        "--decoys_trial_num",
        type=int,
        default=1,
        help="idx of decoys generation trial",
    )
    group.add_argument(
        "--version",
        type=int,
        default=2,
        help="version of trainer and other codes",
    )
    group.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="number of epochs to train"
    )


    # ---------------  Optional hyper-parameters ---------------

    # Efficiency.
    group.add_argument(
        "--checkpoint-activations",
        action="store_true",
        help="checkpoint activation to allow for training "
        "with larger models and sequences. become slow (< 1.5x), save CUDA memory.",
    )
    group.add_argument("--fp16", action="store_true", help="Run model in fp16 mode")
    group.add_argument(
        "--bf16",
        action="store_true",  # only A100 supports it. Not fully tested.
        help="Run model in bf16 mode",
    )
    group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="run optimizer after every gradient-accumulation-steps backwards.",
    )
    group.add_argument(
        "--gradient-clipping",
        type=float,
        default=10.0,
        help="",
    )
    group.add_argument(
        "--epochs", type=int, default=None, help="number of train epochs"
    )
    group.add_argument(
        "--save_interval", type=int, default=None, help="number of save_interval"
    )
    group.add_argument("--log_interval", type=int, default=100, help="report interval")
    
    # group.add_argument("--deepspeed_config", default='/nfs_beijing_ai/jinxian/rama-scoring1.3.0/deepspeed_config.json', help="deepspeed_config")

    group.add_argument(
        "--save-args",
        action="store_true",
        help="save args corresponding to the experiment-name",
    )

    group.add_argument(
        "--model_parallel_size",
        type=int,
        default=1,
        help="model_parallel_size",
    )

    group.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="weight decay coefficient for L2 regularization",
    )

    # model checkpointing
    group.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output directory to save checkpoints to.",
    )

    group.add_argument(
        "--save-method",
        type=str,
        default="model-n-optim",
        choices=["model-only", "model-n-optim"],
        help="Whether save optim params",
    )

    group.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to a directory containing a model checkpoint.",
    )

    group.add_argument(
        "--load-method",
        nargs="+",
        default='model-n-optim',
        choices=["pretrained", "model-only", "model-n-optim"],
        help="Methods to handle checkpoints during loading",
    )

    group.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="number of iterations between saves",
    )
    group.add_argument(
        "--no-save-rng", action="store_true", help="Do not save current rng state."
    )

    group.add_argument(
        "--no-load-rng",
        action="store_true",
        help="Do not load rng state when loading checkpoint.",
    )

    group.add_argument(
        "--resume-dataloader",
        action="store_true",
        help="Resume the dataloader when resuming training. ",
    )

    # distributed training related, don't use them.
    group.add_argument(
        "--distributed-backend",
        default="nccl",
        help="which backend to use for distributed " "training. One of [gloo, nccl]",
    )
    group.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher",
    )
    group.add_argument(
        "--local-rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher",
    )
    group.add_argument("--device", type=int, default=-1)

    # exit, for testing the first period of a long training
    group.add_argument(
        "--exit-interval",
        type=int,
        default=None,
        help="Exit the program after this many new iterations.",
    )

    group.add_argument(
        "--reproduce",
        default=False,
        action="store_true",
        help="reproduce all the results",
    )

    return parser


def add_evaluation_args(parser):
    """Run add_evaluation_args method."""
    # code.
    """Evaluation arguments."""

    group = parser.add_argument_group("validation", "validation configurations")

    group.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Data Loader batch size for evaluation datasets."
        "Defaults to `--batch-size`",
    )
    group.add_argument(
        "--eval-iters",
        type=int,
        default=None,
        help="number of iterations to run for evaluation" "validation/test for",
    )
    group.add_argument(
        "--eval_interval",
        type=int,
        default=None,
        help="interval between running evaluation on validation set",
    )
    group.add_argument(
        "--test_eval_interval",
        type=int,
        default=None,
        help="interval between running evaluation on test set",
    )
    group.add_argument(
        "--strict-eval",
        action="store_true",
        help="won't enlarge or randomly map eval-data, and eval full eval-data.",
    )
    return parser


def add_lr_scheduler_args(parser):
    """Run add_lr_scheduler_args method."""
    # code.
    group = parser.add_argument_group("scheduler", "lr scheduler")
    group.add_argument(
        "--max_lr", type=float, default=1e-4, help="Maximum learning rate"
    )
    group.add_argument(
        "--start_decay_after_n_steps",
        type=int,
        default=5e4,
        help="Start decay after N steps",
    )
    group.add_argument("--decay_factor", type=float, default=0.98, help="Decay factor")
    group.add_argument(
        "--warmup_no_steps", type=int, default=1e3, help="Warm-up number of steps"
    )
    group.add_argument(
        "--decay_every_n_steps", type=int, default=5e4, help="Decay every N steps"
    )
    group.add_argument(
        "--base_lr", type=float, default=0.0, help="Base learning rate, warmup starts from this lr."
    )
    return parser


def add_fasten_args(parser):
    """Run add_fasten_args method."""
    # code.
    group = parser.add_argument_group("fast ops", "fasten")
    group.add_argument(
        "--use_fastfold_optimize",
        default=False,
        action="store_true",
        help="enable fastfold optimize kernel",
    )
    group.add_argument(
        "--use_fast_ops",
        default=False,
        action="store_true",
        help="enable fast layernorm",
    )
    parser.add_argument(
        "--use_fast_kernel",
        default=False,
        action="store_true",
        help="enable fast optimize kernel",
    )
    group.add_argument(
        "--use_fused_adam",
        default=False,
        action="store_true",
        help="use fusedadam instead of torch adam",
    )
    group.add_argument(
        "--use_fused_softmax",
        default=False,
        action="store_true",
        help="use fused softmax instead of torch softmax",
    )
    group.add_argument(
        "--use_flash_attention",
        default=False,
        action="store_true",
        help="enable flash attention optimize kernel",
    )
    parser.add_argument(
        "--use_emsa_flash_attention",
        default=False,
        action="store_true",
        help="enable extraMSA flash attention optimize kernel",
    )
    group.add_argument(
        "--global_chunk_size",
        type=int,
        default=None,
        help="""Number of chunk size in all attention""",
    )
    group.add_argument(
        "--extraMSA_chunk_size",
        type=int,
        default=None,
        help="""Number of chunk in extraMSA attention""",
    )
    group.add_argument(
        "--auto_chunk",
        default=False,
        action="store_true",
        help="auto chunk size in attention override global_chunk_size and extraMSA_chunk_size",
    )
    return parser


def initialize(arg_provider=None):
    """Run initialize method."""
    # code.
    parser = argparse.ArgumentParser(description="AbNbScore")
    parser = add_training_args(parser)
    parser = add_evaluation_args(parser)
    parser = add_lr_scheduler_args(parser)
    parser = add_fasten_args(parser)
    if arg_provider is not None:
        arg_provider(parser)

    add_summary_writer_args(parser)
    group = parser.add_argument_group("Train framework", "configurations")
    group.add_argument("--deepspeed-init-dist", action="store_true")

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    initialize_distributed(args)
    set_random_seed(args.seed)
    return args


def init_deepspeed_config(args):
    """Run init_deepspeed_config method."""
    # code.
    if args.mode != "inference":  # training with deepspeed
        args.deepspeed = True
        if args.deepspeed_config is None:  # not specified
            args.deepspeed_config = os.path.join(
                os.path.dirname(__file__),
                "configs/deepspeed",
                f"deepspeed_zero{args.zero_stage}.json",
            )
            override_deepspeed_config = True
        else:
            override_deepspeed_config = False

    if args.zero_stage > 0:
        assert args.fp16 or args.bf16, "deepspeed zero only support half"

    if args.deepspeed:
        if args.checkpoint_activations:
            args.deepspeed_activation_checkpointing = True
        else:
            args.deepspeed_activation_checkpointing = False
        if args.deepspeed_config is not None:
            with open(args.deepspeed_config) as file:
                deepspeed_config = json.load(file)

        # update: we can choose several deepspeed configs to modify in .sh, and fix other configs.
        if args.override_some_deepspeed_config:
            import re
            override_args = re.split(",",args.override_some_deepspeed_config)
            for arg in override_args:
                if arg == "gradient_accumulation_steps":
                    deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
                elif arg == "gradient_clipping":
                    deepspeed_config["gradient_clipping"] = args.max_grad_norm
                else:
                    raise RuntimeError(f"unknown deepspeed config {arg}")


        if override_deepspeed_config:  # not specify deepspeed_config, use args
            if args.fp16:
                deepspeed_config["fp16"]["enabled"] = True
            else:
                deepspeed_config["fp16"]["enabled"] = False
            deepspeed_config["gradient_clipping"] = args.gradient_clipping
            deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size
            deepspeed_config[
                "gradient_accumulation_steps"
            ] = args.gradient_accumulation_steps
            optimizer_params_config = deepspeed_config["optimizer"]["params"]
            optimizer_params_config["lr"] = args.lr
            optimizer_params_config["weight_decay"] = args.weight_decay
        else:  # override args with values in deepspeed_config
            logger.info("Will override arguments with manually specified deepspeed_config!")
            if "fp16" in deepspeed_config and deepspeed_config["fp16"]["enabled"]:
                args.fp16 = True
            else:
                args.fp16 = False
            if "train_micro_batch_size_per_gpu" in deepspeed_config:
                args.batch_size = deepspeed_config["train_micro_batch_size_per_gpu"]
            if "gradient_accumulation_steps" in deepspeed_config:
                args.gradient_accumulation_steps = deepspeed_config[
                    "gradient_accumulation_steps"
                ]
            else:
                args.gradient_accumulation_steps = None
            if "gradient_clipping" in deepspeed_config:
                args.gradient_clipping = deepspeed_config["gradient_clipping"]
            if "optimizer" in deepspeed_config:
                optimizer_params_config = deepspeed_config["optimizer"].get(
                    "params", {}
                )
                args.lr = optimizer_params_config.get("lr", args.lr)
                args.weight_decay = optimizer_params_config.get(
                    "weight_decay", args.weight_decay
                )

        deepspeed_config["steps_per_print"] = args.log_interval
        args.deepspeed_config = deepspeed_config


def update_args_with_file(args, path):
    """Run update_args_with_file method."""
    # code.
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    # expand relative path
    folder = os.path.dirname(path)
    for k in config:
        # all the relative paths in config are based on the folder
        if k.endswith("_path"):
            config[k] = os.path.join(folder, config[k])
            if args.rank == 0:
                logger.info(f"> parsing relative path {k} in model_config as {config[k]}.")
    args = vars(args)
    for k in list(args.keys()):
        if k in config:
            del args[k]
    args = argparse.Namespace(**config, **args)
    return args


def initialize_distributed(args):
    """Run initialize_distributed method."""
    # code.
    args.cuda = torch.cuda.is_available()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    if args.local_rank is None:
        args.local_rank = int(os.getenv("LOCAL_RANK", "0"))  # torchrun

    if args.device == -1:  # not set manually
        args.device = args.rank % torch.cuda.device_count()
        if args.local_rank is not None:
            args.device = args.local_rank

    # local rank should be consistent with device in DeepSpeed
    if args.local_rank != args.device and args.mode != "inference":
        raise ValueError(
            "LOCAL_RANK (default 0) and args.device inconsistent. "
            "This can only happens in inference mode. "
            "Please use CUDA_VISIBLE_DEVICES=x for single-GPU training. "
        )

    if args.rank == 0:
        logger.info(
            "using world size: {} and model-parallel size: {} ".format(
                args.world_size, args.model_parallel_size
            )
        )
    logger.info(f"""torch.distributed initialized? ={torch.distributed.is_initialized()}""")
    if not torch.distributed.is_initialized():
        # Call the init process
        if getattr(args, "deepspeed_init_dist", False):  # This happens on k8s cluster
            init_method = "deepspeed_init_dist"
            deepspeed.init_distributed()
            # if use torchrun
            args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            args.rank = int(os.getenv("RANK", "0"))
            args.world_size = int(os.getenv("WORLD_SIZE", "1"))
            # if use mpi
        else:  # this happens on local machine mainly
            init_method = "tcp://"
            args.master_ip = os.getenv(
                "MASTER_ADDR", getattr(args, "master_ip", "localhost")
            )
            args.master_port = os.getenv(
                "MASTER_PORT", getattr(args, "master_port", "6000")
            )
            init_method += args.master_ip + ":" + args.master_port
            torch.distributed.init_process_group(
                backend=args.distributed_backend,
                world_size=args.world_size,
                rank=args.rank,
                init_method=init_method,
            )
            

    import socket

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    hostname = socket.gethostname()
    DPRANK = f"DP{mpu.get_data_parallel_rank()}/{mpu.get_data_parallel_world_size()}"
    MPRANK = f"MP{mpu.get_model_parallel_rank()}/{mpu.get_model_parallel_world_size()}"
    if getattr(args, "deepspeed_init_dist", False):
        args.distributed_backend = torch.distributed.get_backend()
    debug_dict = dict(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank,
        local_rank=args.local_rank,
        init_method=init_method,
        device=args.device,
        cuda=args.cuda,
        device_num=torch.cuda.device_count(),
    )
    logger.info(f"{hostname}-{DPRANK}-{MPRANK}-{debug_dict}".replace(" ", ""))

    # the automatic assignment of devices has been moved to arguments.py
    torch.cuda.set_device(args.device)


def set_random_seed(seed):
    """Run set_random_seed method."""
    # code.
    """Set random seed for reproducability."""
    if seed is not None:
        assert seed > 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True  # False
        torch.backends.cuda.matmul.allow_tf32 = (
            False  # if set it to True will be much faster but not accurate
        )
        if (
            not deepspeed_version_ge_0_10 and deepspeed.checkpointing.is_configured()
        ):  # This version is a only a rough number
            mpu.model_parallel_cuda_manual_seed(seed)
