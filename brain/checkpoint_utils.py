import ast
import collections
import contextlib
import inspect
import logging
import os
import re
import time
import traceback
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from fairseq.data import data_utils
from fairseq.dataclass.configs import CheckpointConfig
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
from fairseq.distributed.fully_sharded_data_parallel import FSDP, has_FSDP
from fairseq.file_io import PathManager
from fairseq.models import FairseqDecoder, FairseqEncoder
from omegaconf import DictConfig, OmegaConf, open_dict

logger = logging.getLogger(__name__)


def save_checkpoint(cfg: CheckpointConfig, trainer, epoch_itr, val_loss):
    from fairseq import meters

    # only one worker should attempt to create the required dir
    if trainer.data_parallel_rank == 0:
        os.makedirs(cfg.save_dir, exist_ok=True)

    prev_best = getattr(save_checkpoint, "best", val_loss)
    if val_loss is not None:
        best_function = max if cfg.maximize_best_checkpoint_metric else min
        save_checkpoint.best = best_function(val_loss, prev_best)

    if cfg.no_save:
        return None

    trainer.consolidate_optimizer()  # TODO(SS): do we need this if no_save_optimizer_state

    if not trainer.should_save_checkpoint_on_current_rank:
        if trainer.always_call_state_dict_during_save_checkpoint:
            trainer.state_dict()
        return None

    write_timer = meters.StopwatchMeter()
    write_timer.start()

    epoch = epoch_itr.epoch
    end_of_epoch = epoch_itr.end_of_epoch()
    updates = trainer.get_num_updates()

    logger.info(f"Preparing to save checkpoint for epoch {epoch} @ {updates} updates")

    def is_better(a, b):
        return a >= b if cfg.maximize_best_checkpoint_metric else a <= b

    suffix = trainer.checkpoint_suffix
    checkpoint_conds = collections.OrderedDict()
    checkpoint_conds["checkpoint{}{}.pt".format(epoch, suffix)] = (
        end_of_epoch and not cfg.no_epoch_checkpoints and epoch % cfg.save_interval == 0
    )
    checkpoint_conds["checkpoint_{}_{}{}.pt".format(epoch, updates, suffix)] = (
        not end_of_epoch
        and cfg.save_interval_updates > 0
        and updates % cfg.save_interval_updates == 0
    )
    checkpoint_conds["checkpoint_best{}.pt".format(suffix)] = val_loss is not None and (
        not hasattr(save_checkpoint, "best")
        or is_better(val_loss, save_checkpoint.best)
    )