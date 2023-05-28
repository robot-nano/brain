import math
import torch
import logging

from brain.utils import checkpoints

logger = logging.getLogger(__name__)


@checkpoints.register_checkpoint_hooks
class NoamScheduler:
    def __init__(self, lr_initial, n_warmup_steps, model_size=None):
        self.lr_initial = lr_initial
        self.n_warmup_steps = n_warmup_steps
        self.current_lr = lr_initial
        self.losses = []
        self.n_steps = 0
        self.normalize = n_warmup_steps ** 0.5
        if model_size is not None:
            self.normalize = model_size ** (-0.5)

    def __call__(self, opt):
        self.n_steps += 1

        current_lr = opt.param_groups[0]["lr"]

        lr = self.lr_initial * self._get_lr_scale()

        # Changing the learning rate within the optimizer
        for param_group in opt.param_groups:
            param_group["lr"] = lr

        self.current_lr = current_lr
        return current_lr, lr

    def _get_lr_scale(self):
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return self.normalize * min(
            n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5)
        )

    @checkpoints.mark_as_saver
    def save(self, path):
        data = {"losses": self.losses, "n_steps": self.n_steps}
        torch.save(data, path)

    @checkpoints.mark_as_loader
    def load(self, path, end_of_epoch=False, device=None):
        del end_of_epoch
        del device
        data = torch.load(path)
        self.losses = data["losses"]
        self.n_steps = data["n_steps"]
