import logging
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.distributed as dist
import torch.optim
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.fused_adam import get_fused_adam_class
from omegaconf import II, OmegaConf


logger = logging.getLogger(__name__)

@dataclass
class FairseqAdamConfig(FairseqDataclass):
    adam_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    use_old_adam: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    fp16_adam_stats: bool = field(
        default=False, metadata={"help": "use FP16 stats (with automatic scaling)"}
    )
    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")

class SeqAdam(FairseqOptimizer):
    """Adam optimizer for fairseq.

    Important note: this optimizer corresponds to the "AdamW" variant of
    Adam in its weight decay behavior. As such, it is most closely
    analogous to torch.optim.AdamW from PyTorch.
    """

    def __init__(self, cfg: FairseqAdamConfig, params):
        super().__init__(cfg)
        fused_adam_cls = get_fused_adam_class()
        use_fused_adam = (
            not getattr(cfg, "use_old_adam", False)
            and fused_adam_cls is not None
            and torch.cuda.is_available()
        )
        if getattr(cfg, "tpu", False):
            if self.cfg.fp16_adam_stats:
                raise NotImplementedError("--fp16-adam-stats is only supported on GPU")
            # on TPUs we use the Adam defined here, since it
            # automatically casts gradients to FP32
            self._optimizer = Adam(params, **self.optimizer_config)
        elif use_fused_adam:
            logger.info("using FusedAdam")
            self._optimizer = fused_adam_cls(
                params, use_fp16_stats=self.cfg.fp16_adam_stats, **self.optimizer_config
            )
        else:
            if self.cfg.fp16_adam_stats:
                raise NotImplementedError(
                    "--fp16-adam-stats is only supported with FusedAdamV1"
                )
            self._optimizer = Adam(params, **self.optimizer_config)

    def average_params(self):
    """Reduce Params is only used during BMUF distributed training."""
    state_dict = self.optimizer.state_dict()
    total_gpus = float(dist.get_world_size())
    for _, value in state_dict["state"].items():
        value["exp_avg"] /= total_gpus
        value["exp_avg_sq"] /= total_gpus
        dist.all_reduce(value["exp_avg"], op=dist.ReduceOp.SUM)
        dist.all_reduce(value["exp_avg_sq"], op=dist.ReduceOp.SUM)
