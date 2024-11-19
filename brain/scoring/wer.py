from dataclasses import dataclass, field

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer
from fairseq.scoring.tokenizer import EvaluationTokenizer

class WerScorer(BaseScorer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.reset()
        try:
            import editdistance as ed
        except ImportError:
            raise ImportError("Please install editdistance to use WER scorer")
        self.ed = ed
        self.tokenizer = EvaluationTokenizer(
            tokenizer_type=self.cfg.wer_tokenizer,
            lowercase=self.cfg.wer_lowercase,
            punctuation_removal=self.cfg.wer_remove_punct,
            character_tokenization=self.cfg.wer_char_level,
        )
