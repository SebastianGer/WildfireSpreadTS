from typing import Any

import torch
import torch.nn as nn

from .BaseModel import BaseModel


class PersistenceModel(BaseModel):
    # Persistence model that simply predicts the input fire map as output.
    # use_all_detections: If true, use all detections of the day for the prediction. 
    #                     If false, use only the last detection of the day.

    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        use_all_detections: bool,
        *args: Any,
        **kwargs: Any
    ):

        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            use_doy=False,
            *args,
            **kwargs
        )

        self.use_all_detections = use_all_detections
        # Need a dummy parameter to be able to pass a parameter to the optimizer and sanity checks don't fail
        self.dummy_parameter = nn.Parameter(torch.zeros(1))

    def forward(self, x, doys=None):

        if self.use_all_detections:
            x = x[:, -1, -1, ...]
        else:
            x = x[:, -1, -2, ...]

            x_max = x.amax(dim=[1, 2], keepdim=True)
            # No-fire values are slightly below 0 after normalization. If that is the biggest value,
            # we want to set the maximum to a very large value so that the whole image is predicted as no-fire below.
            x_max[x_max < 0] = 1e10
            x = (x == x_max).float()
        output = (
            x + self.dummy_parameter - self.dummy_parameter
        )
        return output
