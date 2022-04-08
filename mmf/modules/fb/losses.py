# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn
import torch.nn.functional as F
from mmf.common.registry import registry


@registry.register_loss("kd_loss")
class KDLoss(nn.Module):
    """A loss function designed for knowledge distillation

    Args:
        distillation_loss (Dict): A dict containing the name of distillation loss,
            parameters for each different metrics and their weights.
        metric_loss (Dict): A dict containing the name of metric loss
            (classification/retrieval, etc), parameters for each different metrics
            and their weights.

    Example::

        # KDLoss works with config like below where each metric's params and
        # weights are defined
        losses:
        - type: kd_loss
          params:
          distillation_loss:
            loss: mse
            weight: 1
            params: {}
          metric_loss:
            loss: label_smoothing_cross_entropy
            weight: 0.1
            params:
              label_smoothing: 0.1
    """

    def __init__(self, distillation_loss, metric_loss):
        super().__init__()

        self.distill_type = distillation_loss["loss"]
        self.metric_type = metric_loss["loss"]
        self.weights = {
            self.distill_type: distillation_loss["weight"],
            self.metric_type: metric_loss["weight"],
        }
        self.distill_loss_fn = registry.get_loss_class(self.distill_type)(
            **distillation_loss["params"]
        )
        self.metric_loss_fn = registry.get_loss_class(self.metric_type)(
            **metric_loss["params"]
        )
        assert (
            self.distill_loss_fn and self.metric_loss_fn
        ), "Cannot find distillation loss or metric loss"

    def forward(self, sample_list, model_output):
        assert (
            "teacher" in model_output
        ), "Key `teacher` should be present to compute disstillation loss"

        teacher_targets = model_output["teacher"]["scores"]
        if self.distill_type == "softmax_kldiv":
            teacher_targets = F.softmax(teacher_targets, dim=1)

        distill_loss = self.distill_loss_fn(
            {"targets": teacher_targets}, {"scores": model_output["scores"]}
        )
        metric_loss = self.metric_loss_fn(sample_list, model_output)

        loss = (
            self.weights[self.distill_type] * distill_loss
            + self.weights[self.metric_type] * metric_loss
        )
        return loss
