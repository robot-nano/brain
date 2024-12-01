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