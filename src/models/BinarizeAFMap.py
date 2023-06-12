from typing import Any

import torch
import torch.nn as nn

from .BaseModel import BaseModel


class BinarizeAFMap(BaseModel):
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

        # Need a dummy parameter to be able to pass a parameter to the optimizer and sanity checks don't fail
        self.dummy_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # -2.683 is slightly bigger than the transformed version of the no-fire value of 0
        output = (
            (x > -2.683).float().squeeze(axis=1)
            + self.dummy_parameter
            - self.dummy_parameter
        )
        return output
