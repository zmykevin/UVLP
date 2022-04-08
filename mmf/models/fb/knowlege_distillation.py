# Copyright (c) Facebook, Inc. and its affiliates.

import logging
from typing import Dict

import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.utils.file_io import PathManager
from torch import Tensor

logger = logging.getLogger(__name__)


@registry.register_model("kd")
class KnowledgeDistillation(BaseModel):
    """A general knowledge distillation model which contains teacher and student
    models. Teacher model is used to generate soft targets and let student model
    mimic teacher's behavior. Teacher model is built and loaded from a pre-trained
    config and checkpoint, and its parameters are fixed.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

    @classmethod
    def config_path(cls) -> str:
        return "configs/fb/models/kd/defaults.yaml"

    def setup_model(self, config):
        model_name = config.type
        model_class = registry.get_model_class(model_name)

        if model_class is None:
            raise RuntimeError(f"No model registered for name: {model_name}")

        model = model_class(config.params)
        if hasattr(model, "build"):
            model.build()

        # remove model loss if exists
        if hasattr(model, "losses"):
            delattr(model, "losses")

        # Load pretrained checkpoint if provided
        checkpoint_path = config.pretrained_checkpoint
        if checkpoint_path:
            logger.info(f"Loading model parameters from checkpoint `{checkpoint_path}`")
            with PathManager.open(checkpoint_path, "rb") as f:
                state_dict = torch.load(f, map_location=lambda storage, loc: storage)

            model.load_state_dict(state_dict["model"])

        # freeze model parameter
        if config.get("freeze", None):
            for param in model.parameters():
                param.requires_grad = False

        return model

    def build(self):
        logger.info("Setting up teacher model")
        self.teacher_model = self.setup_model(self.config.teacher)

        logger.info("Setting up student model")
        self.student_model = self.setup_model(self.config.student)

    def prepare_samples(
        self, sample_list: Dict[str, Tensor], prefix: str
    ) -> Dict[str, Tensor]:
        """Prepare samples from `sample_list` for teacher and student models.
        `prefix` should be either `teacher` or `student`
        """
        _sample_list = sample_list.copy()
        for key, value in sample_list.items():
            if key.startswith(prefix):
                _sample_list.update(value)

        return _sample_list

    def forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Dict[str, Tensor]]:
        """
        Args:
            sample_list (Dict): the key should have prefix `teacher` and/or `student`
                to extract teacher input features and student input features.
        """
        # sample_list contains text inputs from both teacher and student, they should
        # be separated and sent to different models.
        teacher_sample_list = self.prepare_samples(sample_list, "teacher")
        teacher_outputs = self.teacher_model(teacher_sample_list)

        student_sample_list = self.prepare_samples(sample_list, "student")
        student_outputs = self.student_model(student_sample_list)

        # We return `students_outputs` as `model_outputs`, but should remove `losses`
        # in `student_outputs` so that we can compute distillation loss. Otherwise,
        # MMF will use that `losses`.
        if "losses" in student_outputs:
            del student_outputs["losses"]

        # set `teacher_outputs`
        student_outputs["teacher"] = teacher_outputs
        return student_outputs
