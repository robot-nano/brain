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

class Scorer(object):
    def __init__(self, cfg):
        self.stat = BleuStat()
        self.pad = cfg.pad
        self.eos = cfg.eos
        self.unk = cfg.unk

        try:
            from fairseq import libbleu
        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing libbleu.so. run `pip install --editable .`\n"
            )
            raise e

        self.C = ctypes.cdll.LoadLibrary(libbleu.__file__)

        self.reset()

    def reset(self, one_init=False):
        if one_init:
            self.C.bleu_one_init(ctypes.byref(self.stat))
        else:
            self.C.bleu_zero_init(ctypes.byref(self.stat))

    def add(self, ref, pred):
        if not isinstance(ref, torch.IntTensor):
            raise TypeError("ref must be a torch.IntTensor (got {})".format(type(ref)))
        if not isinstance(pred, torch.IntTensor):
            raise TypeError("pred must be a torch.IntTensor(got {})".format(type(pred)))

        # don't match unknown words
        rref = ref.clone()
        assert not rref.lt(0).any()
        rref[rref.eq(self.unk)] = -999

        rref = rref.contiguous().view(-1)
        pred = pred.contiguous().view(-1)

        self.C.bleu_add(
            ctypes.byref(self.stat),
            ctypes.c_size_t(rref.size(0)),
            ctypes.c_void_p(rref.data_ptr()),
            ctypes.c_size_t(pred.size(0)),
            ctypes.c_void_p(pred.data_ptr()),
            ctypes.c_int(self.pad),
            ctypes.c_int(self.eos),
        )

    def score(self, order=4):
        psum = sum(
            math.log(p) if p > 0 else float("-Inf") for p in self.precision()[:order]
        )
        return self.brevity() * math.exp(psum / order) * 100

    def precision(self):
        def ratio(a, b):
            return a / b if b > 0 else 0

        return [
            ratio(self.stat.match1, self.stat.count1),
            ratio(self.stat.match2, self.stat.count2),
            ratio(self.stat.match3, self.stat.count3),
            ratio(self.stat.match4, self.stat.count4),
        ]