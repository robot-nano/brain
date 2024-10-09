from collections import defaultdict
from itertools import chain

import torch
from omegaconf import DictConfig

from fairseq import optim

from .dynamic_loss_scaler import DynamicLossScaler


class _FP16OptimizerMixin(object):
    def __init__(self, *args, **kwargs):
        # forward __init__ call to the next class in mro(method resolution order)
        super().__init__(*args, **kwargs)
        self._multiply_factor = 1.0

    @property
    def has_flat_params(self):
        return torch.is_tensor(self.fp32_params) or (
            isinstance(self.fp32_params, dict)
            and all(torch.is_tensor(t) for t in self.fp32_params.values())
        )

    @classmethod
    def build_fp32_params(cls, args, params, flatten=True):
        # create FP32 copy of parameters and grads
        if flatten:
            is_pipeline_parallel = getattr(
                args, "pipeline_model_parallel", False
            ) and getattr(args, "distributed_no_spawn", False)
            total_param_size = sum(p.data.numel() for p in params)
            devices = [torch.cuda.current_device()]
            if is_pipeline_parallel:
                devices = list(set(args.pipeline_devices))
            fp32_params = {}
            for device in devices:
                if is_pipeline_parallel:
                    device_param_size = sum(
                        p.data.numel() for p in params if p.device.index == device
                    )
                    device_params = [p for p in params if p.device.index == device]
                else:
                    device_param_size = total_param_size
                    device_params = params
                fp32_params[device] = (
                    device_params[0].new(0).float().new(device_param_size)
                )
                offset = 0
                for p in device_params:
                    numel = p.data.numel()
                    fp32_params[device][offset : offset + numel].copy_(p.data.view(-1))
                    offset += numel
                fp32_params[device] = torch.nn.Parameter(fp32_params[device])
                fp32_params[device].grad = fp32_params[device].data.new(
                    device_param_size
                )
            return fp32_params
        else:
            fp32_params = []
            for p in params:
                p32 = torch.nn.Parameter(p.data.float())
                if hasattr(p, "expert"):
                    p32.expert = True
                elif hasattr(p, "base_expert"):
                    p32.base_expert = True
                p32.grad = torch.zeros_like(p32.data)
                if hasattr(p, "param_group"):
                    p32.param_group = p.param_group
                if hasattr(p, "optim_overrides"):
                    p32.optim_overrides = p.optim_overrides
                fp32_params.append(p32)
            return fp32_params

    def state_dict(self):
        """Return the optimizer's state dict."""
        state_dict = self.fp32_optimizer.state_dict()
        if self.scaler is not None:
            state_dict["loss_scale"] = self.scaler.loss_scale
        return state_dict

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        if "loss_scale" in state_dict and self.scaler is not None:
            self.scaler.loss_scale = state_dict["loss_scale"]
        self.fp32_optimizer.load_state_dict(state_dict, optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves.

        Compared to :func:`fairseq.optim.FairseqOptimizer.backward`, this
        function additionally dynamically scales the loss to avoid gradient
        underflow.
        """
        if self.scaler is not None:
            loss = self.scaler.scale(loss)
        loss.backward()
        self._needs_sync = True

    def _sync_fp16_grads_to_fp32(self):
        if self._needs_sync:
            # copy FP16 grads to FP32
            if self.has_flat_params:
                devices = list(self.fp32_params.keys())
                device_params_dict = defaultdict(list)
                for p in self.fp16_params:
                    if p.requires_grad:
                        device_params_dict[p.device.index].append(p)
                for device in devices:
                    device_params = device_params_dict[device]
                    offset = 0
                    for p in device_params:
                        grad_data = (
                            p.grad.data
                            if p.grad is not None
                            else p.data.new_zeros(p.data.shape)
                        )
                        numel = grad_data.numel()
                        self.fp32_params[device].grad.data[
                            offset : offset + numel
                        ].copy_(grad_data.view(-1))
                        offset += numel
            else:
                for p, p32 in zip(self.fp16_params, self.fp32_params):
                    if not p.requires_grad:
                        continue
                    if p.grad is not None:
                        if p32.grad is None:
                            p32.grad = p.grad.data.float()
                        else:
                            p32.grad.data.copy_(p.grad.data)
                    else:
                        p32.grad = torch.zeros_like(p.data, dtype=torch.float)

            self._needs_sync = False

    def _sync_fp32_params_to_fp16(self):
        # copy FP32 params back into FP16 model
        if self.has_flat_params:
            devices = list(self.fp32_params.keys())
            device_params_dict = defaultdict(list)
            for p in self.fp16_params:
                device_params_dict[p.device.index].append(p)
            for device in devices:
                device_params = device_params_dict[device]
                offset = 0
                for p in device_params:
                    numel = p.data.numel()
                    p.data.copy_(
                        self.fp32_params[device]
                        .data[offset : offset + numel]
                        .view_as(p.data)
                    )
                    offset += numel
        else:
            for p, p32 in zip(self.fp16_params, self.fp32_params):
                if not p.requires_grad:
                    continue
                p.data.copy_(p32.data)

    def _unscale_grads(self):
        self._sync_fp16_grads_to_fp32()
        if (
            # Skip the multiplication if it's a no-op (i.e., if _multiply_factor
            # is 1.0). At the same time, we want to avoid the device-to-host
            # transfer by comparing it to 1.0. Since _multiply_factor starts as
            # a Python float, we roughly assume that if it's a tensor then it's
            # probably not =1.0 anymore and we do the multiplication. Otherwise
            # we can safely check the value without a D2H transfer.
            torch.is_tensor(self._multiply_factor)
            or self._multiply_factor != 1.0
        ):
            self.fp32_optimizer.multiply_grads(self._multiply_factor)
            self._multiply_factor = 1.0