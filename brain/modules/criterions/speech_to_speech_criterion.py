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