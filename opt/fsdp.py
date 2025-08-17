"""Code."""
# FSDP
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    FullOptimStateDictConfig
)
from functools import partial
from torch.distributed.fsdp import  (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        CPUOffload,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
    )
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    CheckpointImpl,
    checkpoint_wrapper,
)
from model.equiformer.equiformer import Equiformer_blocks
import torch
from utils import dist_utils

def wrapper_fsdp(model, args):
    """Run wrapper_fsdp method."""
    # code.
    """FSDP wrapper
    """
    # check_fn = lambda submodule: isinstance(submodule, (MultiHeadCrossDTP, MultiHeadWeightDTP, RadialNN))
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    transformer_auto_wrapper_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Equiformer_blocks,
        },
    )

    fp16_policy = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )

    ignored_modules = []
    for name, module in model.named_modules():
        # Fix Error
        # RuntimeError: Output 0 of ViewBackward0 is a view and its base or another view of
        # its base has been modified inplace. This view is the output of a function that returns
        # multiple views. Such functions do not allow the output views to be modified inplace.
        # You should replace the inplace operation by an out-of-place one.
        if 'embed' in name:
            ignored_modules.append(module)

    model = FSDP(
        model,
        device_id=dist_utils.get_local_rank(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        ignored_modules=ignored_modules,
        auto_wrap_policy=transformer_auto_wrapper_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        #cpu_offload=CPUOffload(offload_params=True),
        #mixed_precision=fp16_policy,
        #use_orig_params=True,
    )
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper) #, check_fn=check_fn)

    return model
