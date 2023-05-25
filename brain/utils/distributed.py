import os
import pdb

import torch
import logging
import torch.distributed as dist

logger = logging.getLogger(__name__)


def run_on_main(
    func,
    args=None,
    kwargs=None,
    post_func=None,
    post_args=None,
    post_kwargs=None,
    run_post_on_main=False,
):
    # Handle the mutable data types' default args:
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if post_args is None:
        post_args = []
    if post_kwargs is None:
        post_kwargs = {}

    if if_main_process():
        # Main comes here
        try:
            func(*args, **kwargs)
        finally:
            ddp_barrier()
    else:
        # Others go here
        ddp_barrier()
    if post_func is not None:
        if run_post_on_main:
            # Just run on every process without any barrier
            post_func(*post_args, **post_kwargs)
        elif not if_main_process():
            # Others go there
            try:
                post_func(*post_args, **post_kwargs)
            finally:
                ddp_barrier()
        else:
            # But main comes here
            ddp_barrier()


def if_main_process():
    if "RANK" in os.environ:
        if os.environ["RANK"] == "":
            return False
        else:
            if int(os.environ["RANK"]) == 0:
                return True
            return False
    return True


def ddp_barrier():
    if dist.is_initialized():
        dist.barrier()


def ddp_init_group(run_opts):
    if run_opts["distributed_launch"]:
        if "RANK" in os.environ is None or os.environ["RANK"] == "":
            raise ValueError(
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.launch [args]\n\t"
                "experiment.py hyperparams.yaml --distributed_launch "
                "--distributed_backend=nccl"
            )
        rank = int(os.environ["RANK"])

        if run_opts["distributed_backend"] == "nccl":
            if not dist.is_nccl_available():
                raise ValueError("NCCL is not supported in your machine.")
        elif run_opts["distributed_backend"] == "gloo":
            if not dist.is_gloo_available():
                raise ValueError("GLOO is not supported in your machine.")
        elif run_opts["distributed_backend"] == "mpi":
            if not dist.is_mpi_available():
                raise ValueError("MPI is not supported in your machine.")
        else:
            logger.info(
                run_opts["distributed_backend"]
                + " communcation protocol doesn't exist."
            )
            raise ValueError(
                run_opts["distributed_backend"]
                + " communcation protocol doesn't exist."
            )
        # rank arg is used to set the right rank of the current process for ddp.
        # if you have 2 servers with 2 gpu:
        # server1:
        #   GPU0: local_rank=device=0, rank=0
        #   GPU1: local_rank=device=1, rank=1
        # server2:
        #   GPU0: local_rank=device=0, rank=2
        #   GPU1: local_rank=device=1, rank=3
        dist.init_process_group(
            backend=run_opts["distributed_backend"], rank=rank
        )
    else:
        logger.info(
            "distributed_launch flag is disabled, "
            "this experiment will be executed without DDP."
        )
        if "local_rank" in run_opts and run_opts["local_rank"] > 0:
            raise ValueError(
                "DDP is disabled, local_rank must not be set.\n"
                "For DDP training, please use --distributed_launch. "
                "For example:\n\tpython -m torch.distributed.launch "
                "experiment.py hyperparams.yaml "
                "--distributed_launch --distributed_backend=nccl"
            )
