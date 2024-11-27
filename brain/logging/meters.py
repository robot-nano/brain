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

