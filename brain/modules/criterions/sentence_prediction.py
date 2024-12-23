import math
from dataclasses import dataclass, field
from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef as _matthews_corrcoef
from scipy.stats import pearsonr, spearmanr

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def matthews_corrcoef(preds, labels):
    # make it consistent with other metrics taking (preds, labels) as input
    mcc = _matthews_corrcoef(labels, preds)
    return mcc

class SentencePredictionConfig(FairseqDataclass):
    classification_head_name: str = field(
        default="sentence_classification_head",
        metadata={"help": "name of the classification head to use"},
    )
    regression_target: bool = field(
        default=False,
    )
    report_mcc: bool = False
    report_acc_and_f1: bool = False
    report_pearson_and_spearman: bool = False

class SentencePredictionCriterion(FairseqCriterion):
    def __init__(self, cfg: SentencePredictionConfig, task):
        super().__init__(task)
        self.classification_head_name = cfg.classification_head_name
        self.regression_target = cfg.regression_target
        self.keep_pred_and_targ = (
            cfg.report_mcc or cfg.report_acc_and_f1 or cfg.report_pearson_and_spearman
        )
        self.report_mcc = cfg.report_mcc
        self.report_acc_and_f1 = cfg.report_acc_and_f1
        self.report_pearson_and_spearman = cfg.report_pearson_and_spearman
        self.label_dict = task.label_dictionary

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and self.classification_head_name in model.classification_heads
        ), "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.classification_head_name,
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            task_loss = F.nll_loss(lprobs, targets, reduction="sum")
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            task_loss = F.mse_loss(logits, targets, reduction="sum")

        logging_output = {}
        loss = task_loss
        # mha & ffn regularization update
        if (
            hasattr(model, "args")
            and hasattr(model.args, "mha_reg_scale_factor")
            and model.args.mha_reg_scale_factor != 0.0
        ):
            mha_reg_loss = model._get_adaptive_head_loss()
            loss += mha_reg_loss
            logging_output.update({"mha_reg_loss": mha_reg_loss})
        if (
            hasattr(model, "args")
            and hasattr(model.args, "ffn_reg_scale_factor")
            and model.args.ffn_reg_scale_factor != 0.0
        ):
            ffn_reg_loss = model._get_adaptive_ffn_loss()
            loss += ffn_reg_loss
            logging_output.update({"ffn_reg_loss": ffn_reg_loss})

        logging_output.update(
            {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample_size,
                "sample_size": sample_size,
            }
        )