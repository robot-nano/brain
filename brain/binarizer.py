import logging
import os
import typing as tp
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool

import torch

from fairseq.data import Dictionary, indexed_dataset
from fairseq.file_chunker_utils import Chunker, find_offsets
from fairseq.file_io import PathManager
from fairseq.tokenizer import tokenize_line

logger = logging.getLogger("binarizer")


@dataclass
class BinarizeSummary:
    """
    Keep track of what's going on in the binarizer
    """

    num_seq: int = 0
    replaced: tp.Optional[Counter] = None
    num_tok: int = 0

    @property
    def num_replaced(self) -> int:
        if self.replaced is None:
            return 0
        return sum(self.replaced.values())

    @property
    def replaced_percent(self) -> float:
        return 100 * self.num_replaced / self.num_tok

    def __str__(self) -> str:
        base = f"{self.num_seq} sents, {self.num_tok} tokens"
        if self.replaced is None:
            return base

        return f"{base}, {self.replaced_percent:.3}% replaced"

    def merge(self, other: "BinarizeSummary"):
        replaced = None
        if self.replaced is not None:
            replaced = self.replaced
        if other.replaced is not None:
            if replaced is None:
                replaced = other.replaced
            else:
                replaced += other.replaced
        self.replaced = replaced
        self.num_seq += other.num_seq
        self.num_tok += other.num_tok


class Binarizer(ABC):
    """
    a binarizer describes how to take a string and build a tensor out of it
    """

    @abstractmethod
    def binarize_line(
        self,
        line: str,
        summary: BinarizeSummary,
    ) -> torch.IntTensor:
        ...


def _worker_prefix(output_prefix: str, worker_id: int):
    return f"{output_prefix}.pt{worker_id}"