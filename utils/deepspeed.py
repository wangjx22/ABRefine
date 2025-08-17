"""Code."""
from typing import Optional, Union
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from packaging import version as pkg_version


from deepspeed.runtime.engine import (
    DeepSpeedEngine,
    DeepSpeedOptimizerCallable,
    DeepSpeedSchedulerCallable,
)

from deepspeed.runtime.pipe.engine import PipelineEngine

from deepspeed.runtime.config import DeepSpeedConfig, DeepSpeedConfigError


from deepspeed.utils import log_dist

from deepspeed.runtime import zero

from deepspeed.pipe import PipelineModule

from deepspeed.git_version_info import version, git_hash, git_branch
from torch.nn.modules import Module
import torch.distributed as dist
from deepspeed.runtime.sparse_tensor import SparseTensor
from deepspeed.moe.utils import is_moe_param


def _parse_version(version_str):
    """Run _parse_version method."""
    # code.
    """Parse a version string and extract the major, minor, and patch versions."""
    ver = pkg_version.parse(version_str)
    return ver.major, ver.minor, ver.micro


# Export version information
__version__ = version
# assert (
#     __version__ == "0.6.1"
# ), f"Only implemented in deepspeed 0.6.1 but currently {__version__}"
__version_major__, __version_minor__, __version_patch__ = _parse_version(__version__)
__git_hash__ = git_hash
__git_branch__ = git_branch

version_ge_0_10 = __version_major__ == 0 and __version_minor__ >= 10


class IgnoreDataTypeCheckDeepspeedEngine(DeepSpeedEngine):
    """Define Class IgnoreDataTypeCheckDeepspeedEngine."""

    @staticmethod
    def _DeepSpeedEngine__check_params(model: Module, dtype: torch.dtype) -> None:
        """Run _DeepSpeedEngine__check_params method."""
        # code.
        if (
            not all(
                param.dtype == dtype
                for param in model.parameters()
                if param.requires_grad
            )
            and dist.get_rank() == 0
        ):
            raise ValueError(
                f"{dtype} is enabled but the following parameters have dtype that is "
                f"not {dtype}: "
                f"{[(n, p.dtype) for n, p in model.named_parameters() if p.dtype != dtype]}"
            )

    def _get_gradients_for_reduction(self):
        """Run _get_gradients_for_reduction method."""
        # code.
        non_expert_grads = []
        expert_grads = {}
        if self.has_moe_layers:
            for key in self.expert_data_parallel_group.keys():
                expert_grads[key] = []

        for param_name, param in self.module.named_parameters():
            if param.grad is None:
                if not param.requires_grad:
                    continue
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                param.grad = torch.zeros(
                    param.size(), dtype=param.dtype, device=param.device
                )

            grad_data = param.grad.data
            if param_name in self.sparse_tensor_module_names or grad_data.is_sparse:
                grad_data = SparseTensor(grad_data)

            if is_moe_param(param):
                expert_grads[param.group_name].append(grad_data)
            else:
                non_expert_grads.append(grad_data)

        return non_expert_grads, expert_grads


def initialize(
    args=None,
    model: torch.nn.Module = None,
    optimizer: Optional[Union[Optimizer, DeepSpeedOptimizerCallable]] = None,
    model_parameters: Optional[torch.nn.Module] = None,
    training_data: Optional[torch.utils.data.Dataset] = None,
    lr_scheduler: Optional[Union[_LRScheduler, DeepSpeedSchedulerCallable]] = None,
    mpu=None,
    dist_init_required: Optional[bool] = None,
    collate_fn=None,
    config=None,
    config_params=None,
):
    """Run initialize method."""
    # code.
    """Initialize the DeepSpeed Engine.

    Arguments:
        args: an object containing local_rank and deepspeed_config fields.
            This is optional if `config` is passed.

        model: Required: nn.module class before apply any wrappers

        optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
            This overrides any optimizer definition in the DeepSpeed json config.

        model_parameters: Optional: An iterable of torch.Tensors or dicts.
            Specifies what Tensors should be optimized.

        training_data: Optional: Dataset of type torch.utils.data.Dataset

        lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and returns a Scheduler object.
            The scheduler object should define a get_lr(), step(), state_dict(), and load_state_dict() methods

        mpu: Optional: A model parallelism unit object that implements
            get_{model,data}_parallel_{rank,group,world_size}()

        dist_init_required: Optional: None will auto-initialize torch.distributed if needed,
            otherwise the user can force it to be initialized or not via boolean.

        collate_fn: Optional: Merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.

        config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
            as an argument instead, as a path or a dictionary.

        config_params: Optional: Same as `config`, kept for backwards compatibility.

    Returns:
        A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

        * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.

        * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
          otherwise ``None``.

        * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
          if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
    """
    log_dist(
        "DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
            __version__, __git_hash__, __git_branch__
        ),
        ranks=[0],
    )
    assert model is not None, "deepspeed.initialize requires a model"

    if not isinstance(model, PipelineModule):
        engine = IgnoreDataTypeCheckDeepspeedEngine(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=dist_init_required,
            collate_fn=collate_fn,
            config=config,
            config_params=config_params,
        )
    else:
        raise NotImplementedError
        assert mpu is None, "mpu must be None with pipeline parallelism"
        engine = PipelineEngine(
            args=args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            training_data=training_data,
            lr_scheduler=lr_scheduler,
            mpu=model.mpu(),
            dist_init_required=dist_init_required,
            collate_fn=collate_fn,
            config=config,
            config_params=config_params,
        )

    return_items = [
        engine,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler,
    ]
    return tuple(return_items)


if version_ge_0_10:
    # Set to torch's distributed package or deepspeed.comm based inside DeepSpeedEngine init
    dist = None

    def initialize(
        args=None,
        model: torch.nn.Module = None,
        optimizer: Optional[Union[Optimizer, DeepSpeedOptimizerCallable]] = None,
        model_parameters: Optional[torch.nn.Module] = None,
        training_data: Optional[torch.utils.data.Dataset] = None,
        lr_scheduler: Optional[Union[_LRScheduler, DeepSpeedSchedulerCallable]] = None,
        mpu=None,
        dist_init_required: Optional[bool] = None,
        collate_fn=None,
        config=None,
        config_params=None,
    ):
        """Run initialize method."""
        # code.
        """Initialize the DeepSpeed Engine.

        Arguments:
            args: an object containing local_rank and deepspeed_config fields.
                This is optional if `config` is passed.

            model: Required: nn.module class before apply any wrappers

            optimizer: Optional: a user defined Optimizer or Callable that returns an Optimizer object.
                This overrides any optimizer definition in the DeepSpeed json config.

            model_parameters: Optional: An iterable of torch.Tensors or dicts.
                Specifies what Tensors should be optimized.

            training_data: Optional: Dataset of type torch.utils.data.Dataset

            lr_scheduler: Optional: Learning Rate Scheduler Object or a Callable that takes an Optimizer and returns a Scheduler object.
                The scheduler object should define a get_lr(), step(), state_dict(), and load_state_dict() methods

            mpu: Optional: A model parallelism unit object that implements
                get_{model,data}_parallel_{rank,group,world_size}()

            dist_init_required: Optional: None will auto-initialize torch distributed if needed,
                otherwise the user can force it to be initialized or not via boolean.

            collate_fn: Optional: Merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.

            config: Optional: Instead of requiring args.deepspeed_config you can pass your deepspeed config
                as an argument instead, as a path or a dictionary.

            config_params: Optional: Same as `config`, kept for backwards compatibility.

        Returns:
            A tuple of ``engine``, ``optimizer``, ``training_dataloader``, ``lr_scheduler``

            * ``engine``: DeepSpeed runtime engine which wraps the client model for distributed training.

            * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
            optimizer is specified in json config else ``None``.

            * ``training_dataloader``: DeepSpeed dataloader if ``training_data`` was supplied,
            otherwise ``None``.

            * ``lr_scheduler``: Wrapped lr scheduler if user ``lr_scheduler`` is passed, or
            if ``lr_scheduler`` specified in JSON configuration. Otherwise ``None``.
        """
        log_dist(
            "DeepSpeed info: version={}, git-hash={}, git-branch={}".format(
                __version__, __git_hash__, __git_branch__
            ),
            ranks=[0],
        )

        # Disable zero.Init context if it's currently enabled
        zero.partition_parameters.shutdown_init_context()

        assert model is not None, "deepspeed.initialize requires a model"

        global dist
        from deepspeed import comm as dist
        from deepspeed.accelerator import get_accelerator
        from deepspeed.utils import logger

        dist_backend = get_accelerator().communication_backend_name()
        dist.init_distributed(
            dist_backend=dist_backend, dist_init_required=dist_init_required
        )

        # Set config using config_params for backwards compat
        if config is None and config_params is not None:
            config = config_params

        # Check for deepscale_config for backwards compat
        if hasattr(args, "deepscale_config") and args.deepscale_config is not None:
            logger.warning(
                "************ --deepscale_config is deprecated, please use --deepspeed_config ************"
            )
            if hasattr(args, "deepspeed_config"):
                assert (
                    args.deepspeed_config is None
                ), "Not sure how to proceed, we were given both a deepscale_config and deepspeed_config"
            args.deepspeed_config = args.deepscale_config
            args.deepscale_config = None

        # Check that we have only one config passed
        if hasattr(args, "deepspeed_config") and args.deepspeed_config is not None:
            assert (
                config is None or config == args.deepspeed_config
            ), "Not sure how to proceed, we were given deepspeed configs in the deepspeed arguments and deepspeed.initialize() function call"
            config = args.deepspeed_config
        assert (
            config is not None
        ), "DeepSpeed requires --deepspeed_config to specify configuration file"

        if not isinstance(model, PipelineModule):
            config_class = DeepSpeedConfig(config, mpu)
            if config_class.hybrid_engine.enabled:
                engine = DeepSpeedHybridEngine(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    model_parameters=model_parameters,
                    training_data=training_data,
                    lr_scheduler=lr_scheduler,
                    mpu=mpu,
                    dist_init_required=dist_init_required,
                    collate_fn=collate_fn,
                    config=config,
                    config_class=config_class,
                )
            else:
                engine = DeepSpeedEngine(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    model_parameters=model_parameters,
                    training_data=training_data,
                    lr_scheduler=lr_scheduler,
                    mpu=mpu,
                    dist_init_required=dist_init_required,
                    collate_fn=collate_fn,
                    config=config,
                    config_class=config_class,
                )
        else:
            assert mpu is None, "mpu must be None with pipeline parallelism"
            mpu = model.mpu()
            config_class = DeepSpeedConfig(config, mpu)
            engine = PipelineEngine(
                args=args,
                model=model,
                optimizer=optimizer,
                model_parameters=model_parameters,
                training_data=training_data,
                lr_scheduler=lr_scheduler,
                mpu=mpu,
                dist_init_required=dist_init_required,
                collate_fn=collate_fn,
                config=config,
                config_class=config_class,
            )

        # Restore zero.Init context if necessary
        zero.partition_parameters.restore_init_context()

        return_items = [
            engine,
            engine.optimizer,
            engine.training_dataloader,
            engine.lr_scheduler,
        ]
        return tuple(return_items)
