from typing import Any

import torch.nn as nn

from .BaseModel import BaseModel


class LogisticRegression(BaseModel):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            *args,
            **kwargs
        )

        # Logistic Regression just consists of a single convolutional layer with a kernel size of 3
        self.model = nn.Conv2d(
            in_channels=n_channels, out_channels=1, kernel_size=3, padding=1
        )
