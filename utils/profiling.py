"""Code."""
import time
import torch

from contextlib import contextmanager

from utils.logger import Logger
from utils.dist_utils import is_rank_0

logger = Logger.logger


@contextmanager
def prof_time(dev_prof_time, time_type):
    """Run prof_time method."""
    # code.
    if dev_prof_time:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        starter.record()
        yield
        ender.record()
        torch.cuda.synchronize()  # WAIT FOR GPU SYNC, waits for the event to complete.
        logger.info(f"{time_type}: {starter.elapsed_time(ender)/1000}s")
    else:
        yield


def prof_memory(device=None):
    """Run prof_memory method."""
    # code.
    """
    Args:
        device (torch.device or int, optional):
            selected device. Returns statistic for the current device,
            given by current_device(), if device is None (default).
    Returns:
        memory_reserved (float, GB):
            the current GPU memory managed by the caching allocator in G bytes for a given device.
        max_memory_allocated (float, GB):
            the maximum GPU memory occupied by tensors in G bytes for a given device.
    """
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
    return memory_reserved, max_memory_allocated


def get_now_time():
    """Run get_now_time method."""
    # code.
    now = time.time()
    ms = int((now - int(now)) * 1000)
    return f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))},{ms:03d}"


def print_rank_0(message, with_time=False):
    """Run print_rank_0 method."""
    # code.
    _message = f"[{get_now_time()}] {message}" if with_time else message
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            logger.info(_message, flush=True)
    else:
        logger.info(_message, flush=True)


class Timers:
    """Define Class Timers."""

    """Group of timers."""

    class Timer:
        """Define Class Timer."""

        """Timer."""

        def __init__(self, name):
            """Run __init__ method."""
            # code.
            self.name_ = name
            self.elapsed_ = 0.0
            self.latest_elapsed_ = 0.0
            self.started_ = False
            self.start_time = time.time()

        def start(self):
            """Run start method."""
            # code.
            """Start the timer."""
            assert not self.started_, "timer has already been started"
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.time()
            self.started_ = True

        def stop(self):
            """Run stop method."""
            # code.
            """Stop the timer."""
            assert self.started_, "timer is not started"
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.latest_elapsed_ = time.time() - self.start_time
            self.elapsed_ += self.latest_elapsed_
            self.started_ = False

        def latest_elapsed(self):
            """Run latest_elapsed method."""
            # code.
            return self.latest_elapsed_ * 1000.0

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

    def log(self, names, normalizer=1.0, reset=True, logger=None, prefix=''):
        """Run log method."""
        # code.
        """Log a group of timers."""
        assert normalizer > 0.0
        string = prefix + "time (ms)"
        for name in names:
            if name not in self.timers:
                continue
            elapsed_time = self.timers[name].elapsed(reset=reset) * 1000.0 / normalizer
            string += " | {}: {:.2f}".format(name, elapsed_time)

        if logger is not None and is_rank_0():
            logger.info(string)
        else:
            print_rank_0(string, with_time=True)


def report_memory(name, logger=None):
    """Run report_memory method."""
    # code.
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    string = name + " memory (MB)"
    string += " | allocated: {}".format(torch.cuda.memory_allocated() / mega_bytes)
    string += " | max allocated: {}".format(
        torch.cuda.max_memory_allocated() / mega_bytes
    )
    string += " | cached: {}".format(torch.cuda.memory_reserved() / mega_bytes)
    string += " | max cached: {}".format(torch.cuda.max_memory_reserved() / mega_bytes)
    if logger is not None and is_rank_0():
        logger.info(string)
    else:
        print_rank_0(string, with_time=True)
