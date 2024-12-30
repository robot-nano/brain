import logging
import math
from collections import OrderedDict

import torch


class MultitaskCriterion:
    def __init__(self, multitask_tasks, rdrop_alpha=0.0):
        self.rdrop_alpha = rdrop_alpha
        self.rdrop_alpha_mtl = rdrop_alpha

        self.multitask_criterion = OrderedDict()
        self.multitask_loss_weight = OrderedDict()
        for task_name, task_obj in multitask_tasks.items():
            if task_obj.args.get_loss_weight(0) == 0:
                logger.info(f"Skip {task_name} loss criterion")
                continue

            rdrop_alpha_task = task_obj.args.rdrop_alpha
            if rdrop_alpha_task is None:
                rdrop_alpha_task = rdrop_alpha
            self.rdrop_alpha_mtl = rdrop_alpha_task
            logger.info(f"rdrop_alpha is set to {rdrop_alpha_task} for {task_name}")

            if task_obj.args.decoder_type == "ctc":
                self.multitask_criterion[task_name] = CtcCriterion(
                    task_obj.args.criterion_cfg,
                    task_obj,
                    rdrop_alpha=rdrop_alpha_task,
                )
            else:
                self.multitask_criterion[
                    task_name
                ] = RdropLabelSmoothedCrossEntropyCriterion(
                    task_obj,
                    task_obj.args.criterion_cfg.sentence_avg,
                    label_smoothing=task_obj.args.criterion_cfg.label_smoothing,
                    rdrop_alpha=rdrop_alpha_task,
                )

    def set_multitask_loss_weight(self, task_name, weight=0.0):
        self.multitask_loss_weight[task_name] = weight

    def get_multitask_loss(self, model, sample, model_out):
        logging_output = {}
        loss = 0.0
        for task_name, task_criterion in self.multitask_criterion.items():
            layer_id = task_criterion.task.args.input_layer
            if isinstance(task_criterion, CtcCriterion):
                if task_criterion.task.args.input_from == "encoder":
                    if len(model_out["encoder_padding_mask"]) > 0:
                        non_padding_mask = ~model_out["encoder_padding_mask"][0]
                        input_lengths = non_padding_mask.long().sum(-1)
                    else:
                        out = model_out["encoder_states"][layer_id]
                        input_lengths = out.new_full(
                            (out.shape[1],), out.shape[0]
                        ).long()

                    task_sample = {
                        "net_input": {
                            "src_tokens": model_out["encoder_states"][
                                layer_id
                            ],  # check batch idx
                            "src_lengths": input_lengths,
                        },
                        "id": sample["id"],
                    }
                else:
                    task_sample = {
                        "net_input": {
                            "src_tokens": model_out["inner_states"][layer_id],
                            "src_lengths": sample["target_lengths"],
                        },
                        "id": sample["id"],
                    }
            else:
                task_sample = {
                    "net_input": {
                        "src_tokens": sample["multitask"][task_name]["net_input"][
                            "prev_output_tokens"
                        ],
                        "encoder_out": {
                            "encoder_out": [model_out["encoder_states"][layer_id]],
                            "encoder_padding_mask": model_out["encoder_padding_mask"],
                        },
                    }
                }

            for key in ["target", "target_lengths", "ntokens"]:
                task_sample[key] = sample["multitask"][task_name][key]

            if task_name == getattr(model, "mt_task_name", None):
                decoder_out = model_out["mt_decoder_out"]
            else:
                decoder_out = None
            task_loss, task_sample_size, task_logging_output = task_criterion(
                model.multitask_decoders[task_name], task_sample, net_output=decoder_out
            )

            loss = loss + self.multitask_loss_weight[task_name] * task_loss
            task_logging_output["loss_weight"] = self.multitask_loss_weight[task_name]
            logging_output[task_name] = task_logging_output
        return loss, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        for task_name in logging_outputs[0]["multitask"].keys():
            # different criterion may return different logging
            # currently only reduce on loss, the most common one
            # ideally the way that losses are reduced should also depend on the task type
            loss_sum = sum(
                log["multitask"][task_name].get("loss", 0) for log in logging_outputs
            )
            sample_size = sum(
                log["multitask"][task_name].get("sample_size", 0)
                for log in logging_outputs
            )

            metrics.log_scalar(
                f"multitask_{task_name}_loss",
                loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )

            loss_weight = logging_outputs[0]["multitask"][task_name].get(
                "loss_weight", 0
            )
            metrics.log_scalar(
                f"multitask_{task_name}_loss_weight",
                loss_weight,
                weight=0,
                priority=250,
            )

class SpeechToUnitMultitaskTaskCriterion(
    RdropLabelSmoothedCrossEntropyCriterion, MultitaskCriterion
):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        rdrop_alpha=0.0,
    ):
        super().__init__(
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            rdrop_alpha,
        )
        MultitaskCriterion.__init__(self, task.multitask_tasks, rdrop_alpha)

    def forward(self, model, sample, reduce=True):
        net_input_concat = {
            "src_tokens": sample["net_input"]["src_tokens"],
            "src_lengths": sample["net_input"]["src_lengths"],
            "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            "tgt_speaker": sample["net_input"].get("tgt_speaker", None),
            "return_all_hiddens": True,
        }

        if self.rdrop_alpha > 0 or self.rdrop_alpha_mtl > 0:
            net_input_concat = duplicate_input(net_input_concat)

        net_output, extra = model(**net_input_concat)
        loss, nll_loss, rdrop_kl_loss = self.compute_loss(
            model, [net_output], sample, reduce=reduce
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, [net_output], sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        if self.rdrop_alpha > 0:
            logging_output["rdrop_kl_loss"] = utils.item(rdrop_kl_loss.data)

        if len(self.multitask_criterion) == 0:
            return loss, sample_size, logging_output

        # multitask
        multitask_loss, multitask_log = self.get_multitask_loss(model, sample, extra)
        loss += multitask_loss
        logging_output["multitask"] = multitask_log

        return loss, sample_size, logging_output