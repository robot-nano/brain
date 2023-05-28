import os.path
import torch
import logging
import csv
import json
import sentencepiece as spm
from brain.dataio.dataio import merge_char
from brain.utils import edit_distance
from brain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


class SentencePiece:
    def __init__(
        self,
        model_dir,
        vocab_size,
        annotation_train=None,
        annotation_read=None,
        model_type="unigram",
        char_format_input=False,
        character_coverage=1.0,
        user_defined_symbols=None,
        max_sentencepiece_length=10,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        split_by_whitespace=True,
        num_sequence=None,
        annotation_list_to_check=None,
        annotation_format="csv",
        text_file=None,
        add_dummy_prefix=True,
    ):
        if model_type not in ["unigram", "bpe", "char"]:
            raise ValueError("model_type must be one of : [unigram, bpe, char]")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        if not isinstance(vocab_size, int):
            raise ValueError("vocab_size must be integer.")

        self.annotation_train = annotation_train
        self.annotation_read = annotation_read
        self.annotation_format = annotation_format

        if self.annotation_train is not None:
            ext = os.path.splitext(self.annotation_train)[1]
            if text_file is None:
                text_file = os.path.join(
                    model_dir,
                    os.path.basename(self.annotation_train).replace(
                        ext, ".txt"
                    ),
                )
            self.text_file = text_file
