import ctypes
import math
import sys
from dataclasses import dataclass, field

import torch
from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer
from fairseq.scoring.tokenizer import EvaluationTokenizer


class BleuStat(ctypes.Structure):
    _fields_ = [
        ("reflen", ctypes.c_size_t),
        ("predlen", ctypes.c_size_t),
        ("match1", ctypes.c_size_t),
        ("count1", ctypes.c_size_t),
        ("match2", ctypes.c_size_t),
        ("count2", ctypes.c_size_t),
        ("match3", ctypes.c_size_t),
        ("count3", ctypes.c_size_t),
        ("match4", ctypes.c_size_t),
        ("count4", ctypes.c_size_t),
    ]