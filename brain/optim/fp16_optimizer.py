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