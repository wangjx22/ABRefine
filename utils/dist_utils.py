"""Code."""
from __future__ import print_function
import os
import time
import torch
import torch.distributed as dist

from utils.logger import Logger

logger = Logger.logger


# Global
mpi_comm = None
distributed = False


def setup_for_distributed(is_master):
    """Run setup_for_distributed method."""
    # code.
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        """Run print method."""
        # code.
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """Run is_dist_avail_and_initialized method."""
    # code.
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Run get_world_size method."""
    # code.
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Run get_rank method."""
    # code.
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    """Run get_local_rank method."""
    # code.
    if torch.distributed.is_initialized():
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


def is_main_process():
    """Run is_main_process method."""
    # code.
    return is_rank_0()


def is_rank_0():
    """Run is_rank_0 method."""
    # code.
    return get_rank() == 0


def is_distributed():
    """Run is_distributed method."""
    # code.
    return distributed


def save_on_master(*args, **kwargs):
    """Run save_on_master method."""
    # code.
    if is_main_process():
        torch.save(*args, **kwargs)


def barrier():
    """Run barrier method."""
    # code.
    if torch.distributed.is_initialized():
        dist.barrier()


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


def init_distributed_mode():
    """Run init_distributed_mode method."""
    # code.
    global distributed
    # torch job
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    # elif hasattr(args, "rank"):  # manual
    #     pass
    else:  # mpi job
        msg = "Not using distributed mode"
        try:
            from mpi4py import MPI
            import subprocess

            global mpi_comm
            mpi_comm = MPI.COMM_WORLD
            world_size = mpi_comm.Get_size()  # new: gives number of ranks in comm
            rank = mpi_comm.Get_rank()
            if world_size > 1:
                master_addr = None
                if rank == 0:
                    hostname_cmd = ["hostname -I"]
                    result = subprocess.check_output(hostname_cmd, shell=True)
                    master_addr = result.decode("utf-8").split()[0]
                master_addr = mpi_comm.bcast(master_addr, root=0)
                # Determine local rank by assuming hostnames are unique
                proc_name = MPI.Get_processor_name()
                all_procs = mpi_comm.allgather(proc_name)
                local_rank = sum([i == proc_name for i in all_procs[:rank]])
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(world_size)
                os.environ["LOCAL_RANK"] = str(local_rank)
                os.environ["MASTER_ADDR"] = master_addr
                os.environ["MASTER_PORT"] = "20500"
                logger.info(
                    "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}".format(
                        os.environ["RANK"],
                        os.environ["LOCAL_RANK"],
                        os.environ["WORLD_SIZE"],
                        os.environ["MASTER_ADDR"],
                        os.environ["MASTER_PORT"],
                    )
                )
            else:
                logger.info(msg)
                distributed = False
                return
        except Exception as e:
            logger.info(e)
            logger.info(
                "**mpi4py is not available, using mpirun will not run distributed mode"
            )
            distributed = False
            return

    distributed = True
    logger.info("| distributed init (rank {})".format(rank))

    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", world_size=world_size, rank=rank
    )
    # it's very dangerous, close
    # setup_for_distributed(rank == 0)
