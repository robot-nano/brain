import math
import warnings
from typing import List

import torch
from torch import nn

try:
    from fairseq import ngram_repeat_block_cuda

    EXTENSION_BUILT = True
except ImportError:
    EXTENSION_BUILT = False

def is_cuda_extension_usable() -> bool:
    """Check whether ngram_repeat_block_cuda is built properly"""
    if not EXTENSION_BUILT or not torch.cuda.is_available():
        return False
    bsz = 2
    tokens = torch.tensor([[4, 4, 3, 2], [1, 2, 3, 4]], dtype=torch.long, device="cuda")
    lprobs = torch.rand((8, 12), device="cuda")
    try:
        outputs = ngram_repeat_block_cuda.forward(tokens, lprobs, bsz, 3, 4, 3)
        outputs = outputs + 4  # This line breaks if the extension is built incorrectly.
        return True
    except RuntimeError:
        warnings.warn(
            "NGramRepeatBlock extension must be rebuilt."
            'Run TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0" python setup.py build_ext --inplace'
        )
        return False