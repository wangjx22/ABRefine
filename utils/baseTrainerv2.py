"""Code."""
import os
import traceback

from abc import abstractmethod
from abc import ABC
import time
import random
from collections import defaultdict
from collections.abc import Iterable
from typing_extensions import override

import numpy as np
import deepspeed
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter

import utils.mpu_utils as mpu
from utils.scoring import get_corr
from utils.model_io import save_checkpoint, get_checkpoint_iteration
from utils.deepspeed import version_ge_0_10 as deepspeed_version_ge_0_10
from utils.logger import Logger, print_rank_0

logger = Logger.logger


class UnifiedSummaryWriter:
    """Define Class UnifiedSummaryWriter."""

    def __init__(self, args) -> None:
        """Run __init__ method."""
        # code.
        self.writers = []
        if getattr(args, "wandb", False):
            import wandb

            assert args.wandb_key != ""
            wandb.login(key=args.wandb_key)
            wandb.init(
                project=args.experiment_name.split("-", 1)[0],
                name=args.wandb_name,
            )
            self.writers.append("wandb")
            self.wandb = wandb

        if getattr(args, "tensorboard", False):
            self.tb = SummaryWriter(
                log_dir=os.path.join(args.save, "tensorboard"),
                purge_step=args.iteration,
            )
            self.writers.append("tb")

    def log_scalar_dict(self, metric_dict, step):
        """Run log_scalar_dict method."""
        # code.
        for writer in self.writers:
            if writer == "wandb":
                self.wandb.log(metric_dict, step=step)
            elif writer == "tb":
                for name in metric_dict:
                    self.tb.add_scalar(name, metric_dict[name], step)


def get_summary_writer(args):
    """Run get_summary_writer method."""
    # code.
    return UnifiedSummaryWriter(args)


class AbstractTrainer(ABC):
    """Define Class AbstractTrainer."""

    """
    This is the abstract class of a trainer.
    1. Methods starts with '_' are not recommended override and use by user;
    """

    def forward_step(self):
        """Run forward_step method."""
        # code.
        """This methods calls on every forward"""
        assert False, "not defined"

    def load_checkpoint(self):
        """Run load_checkpoint method."""
        # code.
        """
        load_checkpoint and set iteration
        """
        assert False, "not defined"

    def create_dataset(self) -> torch.utils.data.Dataset:
        """Run create_dataset method."""
        # code.
        """
        return a dataset object of a torch.utils.data.Dataset
        """
        assert False, "not defined"

    def build_model(self) -> torch.nn.Module:
        """Run build_model method."""
        # code.
        """
        return a model object of a torch.nn.Module,
        you suppose to move it by yourself to cuda if using gpu.
        exp.: model.cuda(torch.cuda.current_device())
        """
        assert False, "not defined"

    def on_train_start(self):
        """Run on_train_start method."""
        # code.
        return "Do nothing"

    def on_before_zero_grad(self):
        """Run on_before_zero_grad method."""
        # code.
        return "Do nothing"

    def on_validation_start(self):
        """Run on_validation_start method."""
        # code.
        return "Do nothing"

    def on_validation_end(self):
        """Run on_validation_end method."""
        # code.
        return "Do nothing"

    def fit(self):
        """Run fit method."""
        # code.
        assert False, "not implemented"

    @abstractmethod
    def _fit_loop(self):
        """Run _fit_loop method."""
        # code.
        assert False, "not implemented"

    @abstractmethod
    def _val_n_log(self):
        """Run _val_n_log method."""
        # code.
        assert False, "not implemented"
        _val_loop()

    @abstractmethod
    def _val_loop(self):
        """Run _val_loop method."""
        # code.
        assert False, "not implemented"

    @abstractmethod
    def _make_loaders(self):
        """Run _make_loaders method."""
        # code.
        assert False, "not implemented"

    @abstractmethod
    def _training_step(self):
        """Run _training_step method."""
        # code.
        assert False, "not implemented"


def is_iterable(obj):
    """Run is_iterable method."""
    # code.
    return hasattr(obj, "__iter__") or isinstance(obj, Iterable)


def set_random_seed(seed=42):
    """Run set_random_seed method."""
    # code.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_rng_state():
    """Run print_rng_state method."""
    # code.
    print("Numpy RNG state:", np.random.get_state()[1][:10])  # 打印前5个状态元素
    print("Python RNG state:", random.getstate()[1][:10])    # 打印前5个状态元素
    print("Torch RNG state:", list(torch.get_rng_state().numpy()[:10]))    # 打印前5个状态元素


class BaseTrainer(AbstractTrainer):
    """Define Class BaseTrianer."""

    def __init__(self, args) -> None:
        """Run __init__ method."""
        # code.
        super().__init__()
        self.timers = Timers()
        self.args = args

    @override
    def fit(self):
        """Run fit method."""
        # code.
        """Main training process."""
        args = self.args
        timers = self.timers
        # Experiment Name
        if args.load:  # continue training
            args.experiment_name = os.path.basename(os.path.normpath(args.load))

        # DataLoader
        timers("make_loaders").start()
        logger.info(f"make_loaders start")
        train_data, val_datas, test_datas, fast_warm_up_data = self._make_loaders()

        timers("make_loaders").stop()
        logger.info(f"make_loaders takes {timers('make_loaders').elapsed_} sec")

        if args.epochs:
            args.train_iters = len(train_data)      # train data is a dataloader. len(train_data) will return train_size % batch_size
            if fast_warm_up_data is not None:
                args.fast_warm_up_iters = len(fast_warm_up_data)
            if args.eval_interval is None:
                args.eval_interval = len(train_data) // args.epochs
            if args.save_interval is None:
                args.save_interval = len(train_data) // args.epochs

        # Build model
        timers("Build model").start()
        logger.info(f"Build model start")
        model = self.build_model()
        self._model = model
        timers("Build model").stop()
        logger.info(f"Build model takes {timers('Build model').elapsed_} sec")

        # Config model IO
        timers("Config model IO").start()
        logger.info(f"Config model IO start")
        self.load_checkpoint()
        timers("Config model IO").stop()
        logger.info(f"Config model IO takes {timers('Config model IO').elapsed_} sec")
        if args.save:
            args.save = os.path.join(args.save, args.experiment_name)
            os.system(f"mkdir -p {args.save}")

        # Optimization related things
        timers("Optimization related things").start()
        logger.info(f"Optimization related things start")
        model, optimizer = setup_model_untrainable_params_and_optimizer(args, model)
        self.model = model
        self.optimizer = optimizer
        timers("Optimization related things").stop()
        logger.info(
            f"Optimization related things takes {timers('Optimization related things').elapsed_} sec"
        )

        self.after_deepspeed_initialize()

        # initialize lr scheduler
        lr_scheduler = get_learning_rate_scheduler(optimizer, args)
        self.lr_scheduler = lr_scheduler

        for param_group in optimizer.param_groups:
            param_group["lr"] *= mpu.get_data_parallel_world_size()
            logger.info(f"lr={param_group['lr']}")

        # Load from deepspeed ckpt
        timers("Load deepspeed ckpt").start()
        logger.info(f"Load deepspeed ckpt start")
        self._load_deepspeed_checkpoint()
        timers("Load deepspeed ckpt").stop()
        logger.info(
            f"Load deepspeed ckpt takes {timers('Load deepspeed ckpt').elapsed_} sec"
        )

        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Total parameters: {total_num}, Trainable parameters: {trainable_num}"
        )

        summary_writer = None
        if torch.distributed.get_rank() == 0:
            print_args(args)
            summary_writer = get_summary_writer(args)
        self.summary_writer = summary_writer

        # check dataset and resume dataloader.
        if args.resume_dataloader:
            if is_iterable(train_data.dataset):
                logger.warning(
                    "Warning: we cannot resume iterable dataloader. skipping..."
                )
            else:
                if train_data is not None:
                    train_data.batch_sampler.start_iter = args.iteration % len(
                        train_data
                    )
                if val_datas is not None:
                    start_iter_val = (
                                             args.train_iters // args.save_interval
                                     ) * args.eval_interval
                    val_datas[0].batch_sampler.start_iter = start_iter_val % len(val_datas[0])

        # training
        iteration = 0
        max_epochs = args.max_epochs
        epoch_now = 0
        while max_epochs > 0:
            if args.do_train:
                self.on_train_start()
                if (args.iteration < args.warmup_no_steps) and (fast_warm_up_data is not None):
                    logger.info(f"current iteration:{args.iteration} < {args.warmup_no_steps}")
                    logger.info(f"running warmup on warmup set...")
                    self._warmup_loop(fast_warm_up_data, val_datas, epoch_now, None)
                    logger.info(f"finish warmup one epoch.")
                else:
                    iteration, skipped = self._fit_loop(train_data, val_datas, epoch_now, test_datas)
            epoch_now += 1
            logger.info(f"rank: {torch.distributed.get_rank(group=mpu.get_data_parallel_group())} epochs: {epoch_now}")
            max_epochs -= 1

        # final save
        if args.save and iteration != 0:
            save_checkpoint(
                iteration,
                model,
                optimizer,
                lr_scheduler,
                args,
                client_state=self.get_client_state(),
            )


    def _warmup_loop(self, fast_warm_up_data, val_datas, epoch_now, test_datas=None):
        """Run _warmup_loop method."""
        # code.
        logger.info(f"Warming up on fast warmup dataset...")
        logger.info(f"Epoch: {epoch_now}")
        logger.info(f"iters per epoch of fast warmup dataset (args.fast_warm_up_iters):{self.args.fast_warm_up_iters}")
        """Train the model."""
        timers = self.timers
        args = self.args
        model = self.model
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        summary_writer = self.summary_writer

        # get train data iterator
        if fast_warm_up_data is not None:
            fast_warm_up_data_iterator = iter(fast_warm_up_data)
        else:
            fast_warm_up_data_iterator = None

        # Turn on training mode which enables dropout.
        model.train()

        # Tracking loss.
        total_loss = 0.0
        total_metrics = defaultdict(list)

        # Iterations.
        self.skipped_iters = 0

        try:
            timers("interval time").start()
        except:
            pass

        err_cnt = 0
        while args.iteration < args.fast_warm_up_iters * (epoch_now + 1):    # current epoch not finished yet

            # update: try-catch to skip the batch which may cause CUDA OOM error.
            if err_cnt > 10:
                raise RuntimeError(f"Continuous encountering OOM error. Stop training.")

            try:
                loss, skipped_iter, metrics = self._training_step(
                    fast_warm_up_data_iterator,
                )
                err_cnt = 0
            except Exception as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Encountering OOM error:")
                    logger.warning(f"{type(e).__name__}")
                    logger.warning(f"{e}")
                    logger.warning(f"skip this batch and retry this iter.")
                    torch.cuda.empty_cache()
                    err_cnt += 1
                    try:
                        timers("forward").stop()
                    except:
                        pass
                    try:
                        timers("backward").stop()
                    except:
                        pass
                    continue
                elif type(e) == StopIteration:
                    timers("forward").stop()
                    timers("batch generator1").stop()
                    fast_warm_up_data_iterator = iter(fast_warm_up_data)
                    logger.info('reroll trn dataloader')
                    continue
                else:
                    logger.warning(f"Exception of type {type(e).__name__} encountered.")
                    logger.warning(f"{e}")
                    logger.warning("Traceback:")
                    traceback.print_exc()
                    raise e

            # except StopIteration:
            #     timers("forward").stop()
            #     timers("batch generator1").stop()
            #     fast_warm_up_data_iterator = iter(fast_warm_up_data)
            #     logger.info('reroll trn dataloader')
            #     continue

            self.skipped_iters += skipped_iter
            args.iteration += 1

            # skip the failed case
            if metrics is None:
                continue

            # Update losses.
            total_loss += loss.data.detach().float()
            for name in metrics:
                if "loss" in name:
                    assert (
                            len(metrics[name].shape) == 0
                    ), f"metrics[{name}] without eval must be scalar, but got {metrics[name].shape}"
                    total_metrics[name].append(
                        metrics[name].data.detach().float().item()
                    )
                else:
                    total_metrics[name].append(
                        metrics[name]
                    )

            # Logging.
            if args.iteration % args.log_interval == 0:
                learning_rate = optimizer.param_groups[0]["lr"]
                avg_lm_loss = total_loss.item() / args.log_interval
                # average img & txt loss
                elapsed_time = timers("interval time").elapsed()
                avg_metrics = {}
                for key in total_metrics:
                    if "loss" not in key:
                        continue
                    avg_metrics[key] = sum(total_metrics[key]) / len(total_metrics[key])

                report_iteration_metrics(
                    summary_writer,
                    optimizer,
                    learning_rate,
                    avg_lm_loss,
                    elapsed_time * 1000.0 / args.log_interval,
                    args.iteration,
                    args.train_iters * args.max_epochs,
                    args,
                    avg_metrics,
                )
                total_loss = 0.0
                total_metrics = defaultdict(list)

                timers.log(
                    [
                        "forward",
                        "backward",
                        "allreduce",
                        "optimizer",
                        "batch generator",
                        "batch generator1",
                        "batch generator2",
                        "data loader",
                    ],
                    normalizer=args.log_interval,
                )
            # Checkpointing
            if (
                    args.save
                    and args.save_interval
                    and args.iteration % args.save_interval == 0
            ):
                save_checkpoint(
                    args.iteration,
                    model,
                    optimizer,
                    lr_scheduler,
                    args,
                    client_state=self.get_client_state(),
                )


            # Evaluation valid
            if (args.eval_interval and \
                args.iteration % args.eval_interval == 0 and \
                args.do_valid and val_datas is not None) or \
                (args.iteration % args.fast_eval_interval == 0 and \
                args.iteration < args.fast_eval_stop_iter ):
                logger.info("Start Evaluation on Valid Set.")

                for i, val_data in enumerate(val_datas):
                    if args.strict_eval or args.eval_iters is None:
                        eval_iters = len(val_data)
                    else:
                        eval_iters = args.eval_iters

                    prefix = f"iteration {args.iteration} val set no. {i} process {eval_iters} "

                    self._val_n_log(
                        prefix,
                        val_data,
                        i,
                        eval_iters,
                        False,
                        step=args.iteration,
                        split="val",
                    )

            # Evaluation test
            if (args.test_eval_interval and \
                args.iteration % args.test_eval_interval == 0 and \
                args.do_test and test_datas is not None) or \
                (args.iteration % args.fast_eval_interval == 0 and \
                args.iteration < args.fast_eval_stop_iter ):
                logger.info("Start Evaluation on Test Set.")

                for i, test_data in enumerate(test_datas):
                    if args.strict_eval or args.eval_iters is None:
                        eval_iters = len(test_data)
                    else:
                        eval_iters = args.eval_iters

                    prefix = f"iteration {args.iteration} test set no. {i} process {eval_iters} "

                    self._test_n_log(
                        prefix,
                        test_data,
                        i,
                        eval_iters,
                        False,
                        step=args.iteration,
                        split="test",
                    )

        return args.iteration, self.skipped_iters

    @override
    def _fit_loop(self, train_data, val_datas, epoch_now, test_datas=None):
        """Run _fit_loop method."""
        # code.
        """Train the model."""
        timers = self.timers
        args = self.args
        model = self.model
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        summary_writer = self.summary_writer

        # get train data iterator
        if train_data is not None:
            train_data_iterator = iter(train_data)
        else:
            train_data_iterator = None

        # Turn on training mode which enables dropout.
        model.train()

        # Tracking loss.
        total_loss = 0.0
        total_metrics = defaultdict(list)

        # Iterations.
        self.skipped_iters = 0

        try:
            timers("interval time").start()
        except:
            pass

        err_cnt = 0
        while args.iteration < args.train_iters * (epoch_now + 1):    # current epoch not finished yet
            # update: try-catch to skip the batch which may cause CUDA OOM error.
            if err_cnt > 10:
                raise RuntimeError(f"Continuous encountering OOM error. Stop training.")

            try:
                loss, skipped_iter, metrics = self._training_step(
                    train_data_iterator,
                )
                err_cnt = 0
            except Exception as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"Encountering OOM error:")
                    logger.warning(f"{type(e).__name__}")
                    logger.warning(f"{e}")
                    logger.warning(f"skip this batch and retry this iter.")
                    torch.cuda.empty_cache()
                    err_cnt += 1
                    try:
                        timers("forward").stop()
                    except:
                        pass
                    try:
                        timers("backward").stop()
                    except:
                        pass
                    continue
                elif type(e) == StopIteration:
                    timers("forward").stop()
                    timers("batch generator1").stop()
                    train_data_iterator = iter(train_data)
                    logger.info('reroll trn dataloader')
                    continue
                else:
                    logger.warning(f"Exception of type {type(e).__name__} encountered.")
                    logger.warning(f"{e}")
                    logger.warning("Traceback:")
                    traceback.print_exc()
                    raise e
            # except StopIteration:
            #     timers("forward").stop()
            #     timers("batch generator1").stop()
            #     train_data_iterator = iter(train_data)
            #     logger.info('reroll trn dataloader')
            #     continue

            self.skipped_iters += skipped_iter
            args.iteration += 1

            # skip the failed case
            if metrics is None:
                continue

            # Update losses.
            total_loss += loss.data.detach().float()
            for name in metrics:
                if "loss" in name:
                    assert (
                            len(metrics[name].shape) == 0
                    ), f"metrics[{name}] without eval must be scalar, but got {metrics[name].shape}"
                    total_metrics[name].append(
                        metrics[name].data.detach().float().item()
                    )
                else:
                    total_metrics[name].append(
                        metrics[name]
                    )

            # Logging.
            if args.iteration % args.log_interval == 0:
                learning_rate = optimizer.param_groups[0]["lr"]
                avg_lm_loss = total_loss.item() / args.log_interval
                # average img & txt loss
                elapsed_time = timers("interval time").elapsed()
                avg_metrics = {}
                for key in total_metrics:
                    if "loss" not in key:
                        continue
                    avg_metrics[key] = sum(total_metrics[key]) / len(total_metrics[key])

                report_iteration_metrics(
                    summary_writer,
                    optimizer,
                    learning_rate,
                    avg_lm_loss,
                    elapsed_time * 1000.0 / args.log_interval,
                    args.iteration,
                    args.train_iters * args.max_epochs,
                    args,
                    avg_metrics,
                )
                total_loss = 0.0
                total_metrics = defaultdict(list)

                timers.log(
                    [
                        "forward",
                        "backward",
                        "allreduce",
                        "optimizer",
                        "batch generator",
                        "batch generator1",
                        "batch generator2",
                        "data loader",
                    ],
                    normalizer=args.log_interval,
                )
            # Checkpointing
            if (
                    args.save
                    and args.save_interval
                    and args.iteration % args.save_interval == 0
            ):
                save_checkpoint(
                    args.iteration,
                    model,
                    optimizer,
                    lr_scheduler,
                    args,
                    client_state=self.get_client_state(),
                )

            # Evaluation valid
            if (args.eval_interval and \
                args.iteration % args.eval_interval == 0 and \
                args.do_valid and val_datas is not None) or \
                (args.iteration % args.fast_eval_interval == 0 and \
                args.iteration < args.fast_eval_stop_iter ):
                logger.info(f"Start Evaluation on Valid Sets.")

                for i, val_data in enumerate(val_datas):
                    if args.strict_eval or args.eval_iters is None:
                        eval_iters = len(val_data)
                    else:
                        eval_iters = args.eval_iters

                    prefix = f"iteration {args.iteration} val set no. {i} process {eval_iters} "

                    self._val_n_log(
                        prefix,
                        val_data,
                        i,
                        eval_iters,
                        False,
                        step=args.iteration,
                        split="val",
                    )

            # Evaluation test
            if (args.test_eval_interval and \
                args.iteration % args.test_eval_interval == 0 and \
                args.do_test and test_datas is not None) or \
                (args.iteration % args.fast_eval_interval == 0 and \
                args.iteration < args.fast_eval_stop_iter ):
                logger.info(f"Start Evaluation on Test Sets.")

                for i, test_data in enumerate(test_datas):
                    if args.strict_eval or args.eval_iters is None:
                        eval_iters = len(test_data)
                    else:
                        eval_iters = args.eval_iters

                    prefix = f"iteration {args.iteration} test set no. {i} process {eval_iters} "

                    self._test_n_log(
                        prefix,
                        test_data,
                        i,
                        eval_iters,
                        False,
                        step=args.iteration,
                        split="test",
                    )

        return args.iteration, self.skipped_iters

    @override
    def _training_step(
            self,
            data_iterator,
            single_step=False,
            **kwargs,
    ):
        """Run _training_step method."""
        # code.

        """Single training step."""
        model = self.model
        optimizer = self.optimizer
        lr_scheduler = self.lr_scheduler
        args = self.args
        timers = self.timers

        loss_total, metrics_total, count = 0.0, {}, 0
        model.train()
        # err_cnt = 0
        while True:
            # Forward model for one step.
            timers("forward").start()

            # # update: try-catch to skip the batch which may cause CUDA OOM error.
            # if err_cnt > 10:
            #     raise RuntimeError(f"Continuous encountering OOM error. Stop training.")

            forward_ret = self.forward_step(data_iterator)
            # try:
            #     forward_ret = self.forward_step(data_iterator)
            #     err_cnt = 0   # clear the error count.
            # except Exception as e:
            #     # if cuda oom, then skip this batch
            #     if "out of memory" in str(e).lower():
            #         logger.warning(f"Encountering OOM error:")
            #         logger.warning(f"{type(e).__name__}")
            #         logger.warning(f"{e}")
            #         logger.warning(f"skip this batch.")
            #         torch.cuda.empty_cache()
            #         err_cnt += 1
            #         timers("forward").stop()
            #         continue
            #     # if other error encountering, log the traceback and raise the error.
            #     else:
            #         logger.warning(f"Exception of type {type(e).__name__} encountered.")
            #         logger.warning(f"{e}")
            #         logger.warning("Traceback:")
            #         traceback.print_exc()
            #         raise e

            timers("forward").stop()
            if forward_ret is None:
                logger.info('trn bad smp skip')
                loss, metrics = torch.zeros((), requires_grad=True).to(torch.cuda.current_device()), {}
            elif isinstance(forward_ret, tuple):
                loss, metrics = forward_ret
            else:
                raise ValueError('do not support line 442')

            # Check nan or inf in forward, preventing it from interfering loss scaler,
            loss_reduced = loss.detach().clone()
            torch.distributed.all_reduce(loss_reduced.data)
            loss_reduced.data = loss_reduced.data / args.world_size

            loss_checker = loss_reduced
            for name in metrics:
                if "loss" in name:
                    metrics[name] = (
                        metrics[name].detach().clone()
                        if isinstance(metrics[name], torch.Tensor)
                        else torch.tensor(metrics[name]).cuda(
                            torch.cuda.current_device()
                        )
                    )
                    torch.distributed.all_reduce(metrics[name].data)
                    metrics[name].data /= args.world_size
                    loss_checker = loss_checker + metrics[name]
            if loss_checker.isnan().any() or loss_checker.isinf().any():
                logger.info(
                    "Skipping backward and optimizer step for nan or inf in forwarding metrics/loss!"
                )
                logger.info(f"loss: {loss}")
                return loss.detach(), 1, None

            # Accumulate the statistics
            loss_total += loss_reduced
            for name in metrics:
                if name not in metrics_total:
                    metrics_total[name] = []
                metrics_total[name].append(metrics[name])

            count += 1

            # Calculate gradients, reduce across processes, and clip.
            timers("backward").start()
            if args.deepspeed:
                model.backward(loss)
            else:
                raise ValueError("Currently, we only support training with deepspeed.")
            timers("backward").stop()

            self.on_before_zero_grad()

            # Update parameters.
            skipped_iter, complete = 0, False
            timers("optimizer").start()
            if args.deepspeed:
                if model.is_gradient_accumulation_boundary():
                    model.step()
                    complete = True
                    if not (args.fp16 and optimizer.overflow):
                        lr_scheduler.step()
                    else:
                        skipped_iter = 1
                else:
                    model.step()
            else:
                raise ValueError("Currently, we only support training with deepspeed.")
            timers("optimizer").stop()
            if complete or single_step:
                break
        loss_total /= count

        new_metrics_total = {}
        for key, value in metrics_total.items():
            if 'loss' in key:
                # new_metrics_total[key] = (sum(value) / len(value)).mean()
                new_metrics_total[key] = torch.cat(value, dim=-1).mean()
            else:
                new_metrics_total[key] = value

        return loss_total, skipped_iter, new_metrics_total

    @override
    def _val_n_log(
            self,
            prefix,
            val_data,
            idx,
            eval_iters,
            has_last,
            split,
            verbose=False,
            step=None,
    ):
        """Run _val_n_log method."""
        # code.
        # get validate data iterator
        if val_data is not None:
            val_data_iterator = iter(val_data)
        else:
            val_data_iterator = None
        #val
        loss, metrics = self._val_loop(
            val_data_iterator,
            eval_iters,
            idx,
            verbose,
            has_last,
            step,
        )

        # log
        if torch.distributed.get_rank(group=mpu.get_data_parallel_group()) == 0:
            report_evaluate_metrics(
                self.summary_writer, prefix, idx, loss, step, metrics, mode=split
            )
        torch.cuda.empty_cache()
        return loss

    def _test_n_log(
            self,
            prefix,
            test_data,
            idx,
            eval_iters,
            has_last,
            split,
            verbose=False,
            step=None,
    ):
        """Run _test_n_log method."""
        # code.

        # get test data iterator
        if test_data is not None:
            test_data_iterator = iter(test_data)
        else:
            test_data_iterator = None

        #val
        loss, metrics = self._test_loop(
            test_data_iterator,
            eval_iters,
            idx,
            verbose,
            has_last,
            step,
        )

        # log
        if torch.distributed.get_rank(group=mpu.get_data_parallel_group()) == 0:
            report_evaluate_metrics(
                self.summary_writer, prefix, idx, loss, step, metrics, mode=split
            )
        torch.cuda.empty_cache()
        return loss

    @override
    def _val_loop(
            self,
            data_iterator,
            eval_iters,
            idx,
            verbose=True,
            has_last=True,
            step=None,
    ):
        """Run _val_loop method."""
        # code.
        """Evaluation."""
        model = self.model
        args = self.args
        timers = self.timers
        # Turn on evaluation mode which disables dropout.
        model.eval()
        rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        total_loss, metrics_total = 0, {}
        last_shape = args.val_last_shape[idx]
        drop_number = args.val_drop_number[idx]
        self.on_validation_start()
        with torch.no_grad():
            iteration = 0

            while iteration < eval_iters:
                iteration += 1

                try:
                    forward_ret = self.forward_step(data_iterator)
                except StopIteration:
                    timers("batch generator1").stop()
                    logger.info('val data iterator endpoint')
                    iteration = eval_iters
                    continue
                if forward_ret is None:
                    loss = torch.zeros((), requires_grad=False).to(torch.cuda.current_device())
                    metrics = {
                        'pdb_fpath': None,
                        'pred': None,
                        "rmsd_ca_95": None,
                        "rmsd_ca": None,
                        "cdr3_rmsd_ca_95": None,
                        "cdr3_rmsd_ca": None,
                    }
                else:
                    loss, metrics = forward_ret

                """when contiguous memory optimizations are enabled, the buffers
                allocated by the optimizations are deallocated during backward pass
                in the absence of backward pass the buffers should be reset after each
                forward pass"""
                if args.deepspeed and args.deepspeed_activation_checkpointing:
                    deepspeed.checkpointing.reset()
                total_loss += loss.data.detach().float().item()
                is_last = (
                    True
                    if iteration == eval_iters
                       and args.strict_eval
                       and len(last_shape) > 0
                    else False
                )
                for name in metrics:
                    torch.distributed.barrier()

                    metrics_gathered = [None for _ in range(args.world_size)]

                    torch.distributed.all_gather_object(metrics_gathered, metrics[name])

                    if rank == 0:

                        gathered_len = (
                            len(metrics_gathered)
                            if not is_last
                            else len(metrics_gathered) - drop_number * getattr(args, "model_parallel_size", 1)
                        )

                        for i in range(gathered_len):
                            if name not in metrics_total:
                                metrics_total[name] = []
                            metrics_total[name].append(
                                metrics_gathered[i]
                            )

        self.on_validation_end()
        # Move model back to the train mode.
        total_loss /= eval_iters
        if rank == 0:
            metrics_total['total_loss'] = total_loss
            metrics = self.handle_metrics_function(metrics_total, step=step, idx=idx)
        else:
            metrics = None

        return total_loss, metrics

    def _test_loop(
            self,
            data_iterator,
            eval_iters,
            idx,
            verbose=True,
            has_last=True,
            step=None,
    ):
        """Run _test_loop method."""
        # code.
        """Evaluation."""
        logger.info(f"Start evaluation on Test set.")
        model = self.model
        args = self.args
        timers = self.timers
        # Turn on evaluation mode which disables dropout.
        model.eval()
        rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
        total_loss, metrics_total = 0, {}
        last_shape = args.test_last_shape[idx]
        drop_number = args.test_drop_number[idx]
        self.on_validation_start()
        with torch.no_grad():
            iteration = 0

            while iteration < eval_iters:
                iteration += 1

                try:
                    forward_ret = self.forward_step(data_iterator)
                except StopIteration:
                    timers("batch generator1").stop()
                    logger.info('val data iterator endpoint')
                    iteration = eval_iters
                    continue
                if forward_ret is None:
                    loss = torch.zeros((), requires_grad=False).to(torch.cuda.current_device())
                    metrics = {
                        'pdb_fpath': None,
                        'pred': None,
                        "rmsd_ca_95": None,
                        "rmsd_ca": None,
                        "cdr3_rmsd_ca_95": None,
                        "cdr3_rmsd_ca": None,
                    }
                else:
                    loss, metrics = forward_ret

                """when contiguous memory optimizations are enabled, the buffers
                allocated by the optimizations are deallocated during backward pass
                in the absence of backward pass the buffers should be reset after each
                forward pass"""
                if args.deepspeed and args.deepspeed_activation_checkpointing:
                    deepspeed.checkpointing.reset()
                total_loss += loss.data.detach().float().item()
                is_last = (
                    True
                    if iteration == eval_iters
                       and args.strict_eval
                       and len(last_shape) > 0
                    else False
                )
                for name in metrics:
                    torch.distributed.barrier()

                    metrics_gathered = [None for _ in range(args.world_size)]

                    torch.distributed.all_gather_object(metrics_gathered, metrics[name])

                    if rank == 0:

                        gathered_len = (
                            len(metrics_gathered)
                            if not is_last
                            else len(metrics_gathered) - drop_number * getattr(args, "model_parallel_size", 1)
                        )

                        for i in range(gathered_len):
                            if name not in metrics_total:
                                metrics_total[name] = []
                            metrics_total[name].append(
                                metrics_gathered[i]
                            )

        self.on_validation_end()
        # Move model back to the train mode.
        total_loss /= eval_iters
        if rank == 0:
            metrics_total['total_loss'] = total_loss
            metrics = self.handle_metrics_function(metrics_total, step=step, idx=idx, mode='test')
        else:
            metrics = None

        return total_loss, metrics

    @override
    def load_checkpoint(self):
        """Run load_checkpoint method."""
        # code.
        logger.info(f"args.load:{self.args.load}")
        if self.args.load:
            self.args.iteration = get_checkpoint_iteration(self.args.load)
        else:
            self.args.iteration = 0

    def preprocess_metrics(self, metrics_total):
        """Run preprocess_metrics method."""
        # code.
        new_preds = []
        new_gts = []
        new_pdbs = []
        for pred, rmsd_ca_95, rmsd_ca, cdr3_rmsd_ca_95, cdr3_rmsd_ca, pdb in zip(
                metrics_total['pred'],
                metrics_total['rmsd_ca_95'],
                metrics_total['rmsd_ca'],
                metrics_total['cdr3_rmsd_ca_95'],
                metrics_total['cdr3_rmsd_ca'],
                metrics_total['pdb_fpath'],
        ):
            if pred is None or rmsd_ca_95 is None or rmsd_ca is None or cdr3_rmsd_ca_95 is None or cdr3_rmsd_ca is None or pdb is None:
                continue
            if isinstance(pred, list):
                for item_pred, item_rmsd_ca_95, item_rmsd_ca, item_cdr3_rmsd_ca_95, item_cdr3_rmsd_ca, item_pdb in zip(
                        pred, rmsd_ca_95, rmsd_ca, cdr3_rmsd_ca_95, cdr3_rmsd_ca, pdb
                ):
                    new_preds.append(item_pred)
                    new_gts.append(
                        {
                            'rmsd_ca_95': item_rmsd_ca_95,
                            'rmsd_ca': item_rmsd_ca,
                            'cdr3_rmsd_ca_95': item_cdr3_rmsd_ca_95,
                            'cdr3_rmsd_ca': item_cdr3_rmsd_ca,
                        }
                    )
                    new_pdbs.append(item_pdb)
            else:
                raise ValueError('unkonw datatype')
        metrics_total['pred'], metrics_total['label'], metrics_total['pdb_fpath'] = new_preds, new_gts, new_pdbs
        return metrics_total

    def handle_metrics_function(self, metrics_total, step=None, idx=None, mode='valid', *args, **kwargs):
        """Run handle_metrics_function method."""
        # code.
        metrics_total = self.preprocess_metrics(metrics_total)

        metrics = get_corr(
            metrics_total['pred'], metrics_total['label'], metrics_total['pdb_fpath'], step, idx, self.args.save, mode
        )
        for key in metrics_total.keys():
            if 'loss' in key:
                if metrics_total[key].__class__ == list:
                    metrics[key] = torch.Tensor(metrics_total[key]).mean().detach().item()
                else:
                    metrics[key] = metrics_total[key]

        metrics['total_loss_ori'] = metrics_total['total_loss']
        metrics['smp_count'] = len(metrics_total['pdb_fpath'])
        return metrics

    @override
    def _make_loaders(self):
        """Run _make_loaders method."""
        # code.
        """makes training/val/test"""
        raise NotImplementedError

    def _load_deepspeed_checkpoint(self):
        """Run _load_deepspeed_checkpoint method."""
        # code.
        args = self.args
        if (
                "model-only" in args.load_method or "model-n-optim" in args.load_method
        ) and os.path.exists(os.path.join(args.save, "latest")):
            logger.info(f"Loading ds ckpt load_method = {args.load_method}")
            load_path, client_states = self.model.load_checkpoint(
                load_dir=args.save,
                tag=None,
                load_module_strict=True,
                load_optimizer_states=not ("model-only" in args.load_method),
                load_lr_scheduler_states=True,
                load_module_only=("model-only" in args.load_method),
            )
            self.args.iteration = int(open(os.path.join(args.save, "latest")).read())
            logger.info(f"Successfully loaded load_path={load_path}")
            args.resume_dataloader = True
            logger.info(f"Setting args.resume_dataloader to True")

            self.set_client_state(client_states)
            del client_states

    def set_client_state(self, client_states):
        """Run set_client_state method."""
        # code.
        if not self.args.no_load_rng:  # rng states.
            try:
                random.setstate(client_states["random_rng_state"])
                np.random.set_state(client_states["np_rng_state"])
                torch.set_rng_state(client_states["torch_rng_state"])
                torch.cuda.set_rng_state(client_states["cuda_rng_state"])
                mpu.get_cuda_rng_tracker().set_states(
                    client_states["rng_tracker_states"]
                )

                if self.lr_scheduler is not None:
                    self.lr_scheduler.load_state_dict(
                        client_states["client_lr_scheduler"]
                    )
            except KeyError:
                print_rank_0(
                    "Unable to load optimizer from checkpoint, exiting. "
                    "Specify --no-load-rng to prevent "
                    "attempting to load the random "
                    "state."
                )
                exit()
        return True

    def get_client_state(self):
        """Run get_client_state method."""
        # code.
        state_dict = {}
        state_dict["iteration"] = self.args.iteration
        if self.lr_scheduler is not None:
            state_dict["client_lr_scheduler"] = self.lr_scheduler.state_dict()
        # rng states.
        if not self.args.no_save_rng:
            state_dict["random_rng_state"] = random.getstate()
            state_dict["np_rng_state"] = np.random.get_state()
            state_dict["torch_rng_state"] = torch.get_rng_state()
            state_dict["cuda_rng_state"] = torch.cuda.get_rng_state()
            state_dict["rng_tracker_states"] = mpu.get_cuda_rng_tracker().get_states()
        return state_dict


def get_params_for_weight_decay_optimization(module):
    """Run get_params_for_weight_decay_optimization method."""
    # code.
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}

    for module_ in module.modules():
        weight_decay_params["params"].extend(
            [
                p
                for n, p in list(module_._parameters.items())
                if p is not None and n != "bias" and p.requires_grad
            ]
        )
        no_weight_decay_params["params"].extend(
            [
                p
                for n, p in list(module_._parameters.items())
                if p is not None and n == "bias" and p.requires_grad
            ]
        )

    if len(weight_decay_params["params"]) == 0:
        return (no_weight_decay_params,)
    elif len(no_weight_decay_params["params"]) == 0:
        return (weight_decay_params,)

    return weight_decay_params, no_weight_decay_params

    def after_deepspeed_initialize(self):
        return None


def get_optimizer_param_groups(model):
    """Run get_optimizer_param_groups method."""
    # code.
    # Build parameter groups (weight decay and non-decay).
    if hasattr(model, "module"):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)  # TODO move to here
    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group["params"]:
            if not hasattr(param, "model_parallel"):
                param.model_parallel = False
    return param_groups


def setup_model_untrainable_params_and_optimizer(args, model):
    """Run setup_model_untrainable_params_and_optimizer method."""
    # code.
    if hasattr(model, "disable_untrainable_params"):
        model.disable_untrainable_params()  # mark trainable params

    param_groups = get_optimizer_param_groups(model)
    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        init_fn = deepspeed.initialize
        if deepspeed_version_ge_0_10:
            model, optimizer, _, _ = init_fn(
                model=model,
                model_parameters=param_groups,
                args=args,
                mpu=mpu,
                dist_init_required=False,
            )
        else:
            model, optimizer, _, _ = init_fn(
                model=model,
                model_parameters=param_groups,
                args=args,
                mpu=mpu,
                dist_init_required=False,
                config_params=args.deepspeed_config,
            )
    else:
        raise ValueError("Currently, we only support training with deepspeed.")
    logger.info(f"Initialized optimizer {type(optimizer)} {optimizer}")
    return model, optimizer


def report_iteration_metrics(
        summary_writer,
        optimizer,
        lr,
        loss,
        elapsed_time,
        step,
        total_step,
        args,
        avg_metrics,
):
    """Run report_iteration_metrics method."""
    # code.
    log_string = " iteration {:8d}/{:8d} |".format(step, total_step)
    log_string += " elapsed time per iteration (ms): {:.1f} |".format(elapsed_time)
    log_string += " learning rate {:.3E} |".format(lr)
    log_string += " total loss {:.6E} |".format(loss)
    for key in avg_metrics:
        log_string += " {} {:.6E} |".format(key, avg_metrics[key])
    if args.fp16:
        log_string += " loss scale {:.1f} |".format(
            optimizer.cur_scale if args.deepspeed else optimizer.loss_scale
        )
    print_rank_0(log_string)
    if summary_writer is not None:
        log_dict = {f"Train/{k}": v for k, v in avg_metrics.items()}
        log_dict["Train/lr"] = lr
        log_dict["Train/loss"] = loss
        log_dict["Train/elapsed_time"] = elapsed_time
        summary_writer.log_scalar_dict(log_dict, step)


def get_learning_rate_scheduler(optimizer, args):
    """Run get_learning_rate_scheduler method."""
    # code.
    """Build the learning rate scheduler."""
    from utils.lr_schedulers import LRScheduler

    lr_scheduler_config = {
        "max_lr": args.max_lr,
        "start_decay_after_n_steps": args.start_decay_after_n_steps,
        "decay_factor": args.decay_factor,
        "warmup_no_steps": args.warmup_no_steps,
        "decay_every_n_steps": args.decay_every_n_steps,
        "base_lr": args.base_lr,
    }

    logger.info(f"Initialize lr_scheduler with kwargs {lr_scheduler_config}")
    return LRScheduler(optimizer=optimizer, **lr_scheduler_config)


def report_evaluate_metrics(summary_writer, prefix, idx, loss, step, avg_metrics, mode='val'):
    """Run report_evaluate_metrics method."""
    # code.
    string = f"the {idx} evaluation loss at {prefix} | "
    string += "loss: {:.6E} | ".format(loss)
    for key in avg_metrics:
        string += " {} {:.6E} |".format(key, avg_metrics[key])

    string += f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB"
    string += f"Max GPU memory usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f}GB"
    length = len(string) + 1
    print_rank_0("-" * 100)
    print_rank_0("-" * length)
    print_rank_0(string)
    print_rank_0("-" * length)
    if summary_writer is not None:
        log_dict = {f"{mode}_{idx}/{k}": v for k, v in avg_metrics.items()}
        log_dict[f"{mode}_{idx}/loss"] = loss
        summary_writer.log_scalar_dict(log_dict, step)


SUMMARY_WRITER_DIR_NAME = "runs"


def print_args(args):
    """Run print_args method."""
    # code.
    """Print arguments."""

    logger.info("arguments:")
    for arg in vars(args):
        dots = "." * (29 - len(arg))
        logger.info("  {} {} {}".format(arg, dots, getattr(args, arg)))
    if args.save_args:
        os.makedirs(
            os.path.join(args.summary_dir, SUMMARY_WRITER_DIR_NAME), exist_ok=True
        )
        with open(
                os.path.join(
                    args.summary_dir, SUMMARY_WRITER_DIR_NAME, args.experiment_name + ".txt"
                ),
                "w",
        ) as f:
            for arg in vars(args):
                dots = "." * (29 - len(arg))
                f.write("  {} {} {}\n".format(arg, dots, getattr(args, arg)))


class Timers:
    """Define Timers Class."""

    """Group of timers."""

    class Timer:
        """Define Timer Class."""

        """Timer."""

        def __init__(self, name):
            """Run __init__ method."""
            # code.
            self.name_ = name
            self.elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Run start method."""
            # code.
            """Start the timer."""
            assert not self.started_, f"timer {self.name_} has already been started"
            torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Run stop method."""
            # code.
            """Stop the timer."""
            assert self.started_, f"timer {self.name_} is not started"
            torch.cuda.synchronize()
            self.elapsed_ += time.time() - self.start_time
            self.started_ = False

        def reset(self):
            """Run reset method."""
            # code.
            """Reset timer."""
            self.elapsed_ = 0.0
            self.started_ = False

        def elapsed(self, reset=True):
            """Run elapsed method."""
            # code.
            """Calculate the elapsed time."""
            started_ = self.started_
            # If the timing in progress, end it first.
            if self.started_:
                self.stop()
            # Get the elapsed time.
            elapsed_ = self.elapsed_
            # Reset the elapsed time
            if reset:
                self.reset()
            # If timing was in progress, set it back.
            if started_:
                self.start()
            return elapsed_

    def __init__(self):
        """Run __init__ method."""
        # code.
        self.timers = {}

    def __call__(self, name):
        """Run __call__ method."""
        # code.
        if name not in self.timers:
            self.timers[name] = self.Timer(name)
        return self.timers[name]

    def log(self, names, normalizer=1.0, reset=True):
        """Run log method."""
        # code.
        """Log a group of timers."""
        assert normalizer > 0.0
        string = "time (ms)"
        for name in names:
            if name not in self.timers:
                continue
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f}".format(name, elapsed_time)
        print_rank_0(string)
