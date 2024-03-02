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