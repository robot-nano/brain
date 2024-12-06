import bisect
import time
from collections import OrderedDict
from typing import Dict, Optional

try:
    import torch

    def type_as(a, b):
        if torch.is_tensor(a) and torch.is_tensor(b):
            return a.to(b)
        else:
            return a

except ImportError:
    torch = None

    def type_as(a, b):
        return a


try:
    import numpy as np
except ImportError:
    np = None


class Meter(object):
    """Base class for Meters."""

    def __init__(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def reset(self):
        raise NotImplementedError

    @property
    def smoothed_value(self) -> float:
        """Smoothed value used for logging."""
        raise NotImplementedError

class AverageMeter(Meter):
    """Computes and stores the average and current value"""

    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.reset()

    def reset(self):
        self.val = None  # most recent update
        self.sum = 0  # sum from all updates
        self.count = 0  # total n from all updates

    def update(self, val, n=1):
        if val is not None:
            self.val = val
            if n > 0:
                self.sum = type_as(self.sum, val) + (val * n)
                self.count = type_as(self.count, n) + n

    def state_dict(self):
        return {
            "val": self.val,
            "sum": self.sum,
            "count": self.count,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        self.val = state_dict["val"]
        self.sum = state_dict["sum"]
        self.count = state_dict["count"]
        self.round = state_dict.get("round", None)

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else self.val

    @property
    def smoothed_value(self) -> float:
        val = self.avg
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val


class SumMeter(Meter):
    """Computes and stores the sum"""

    def __init__(self, round: Optional[int] = None):
        self.round = round
        self.reset()

    def reset(self):
        self.sum = 0  # sum from all updates

    def update(self, val):
        if val is not None:
            self.sum = type_as(self.sum, val) + val

    def state_dict(self):
        return {
            "sum": self.sum,
            "round": self.round,
        }

    def load_state_dict(self, state_dict):
        self.sum = state_dict["sum"]
        self.round = state_dict.get("round", None)

    @property
    def smoothed_value(self) -> float:
        val = self.sum
        if self.round is not None and val is not None:
            val = safe_round(val, self.round)
        return val