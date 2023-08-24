from typing import Any, Tuple

import torch.nn as nn

from .BaseModel import BaseModel
from .utae_paps_models.convlstm import ConvLSTM, ConvLSTM_Seg


class ConvLSTM_Seg_multi_layers(ConvLSTM_Seg):
    """_summary_ ConvLSTM class where the number of layers can be set. 
    """
    def __init__(
        self,
        num_classes,
        input_size,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        pad_value=0,
    ):
        super(ConvLSTM_Seg, self).__init__()
        self.convlstm_encoder = ConvLSTM(
            input_dim=input_dim,
            input_size=input_size,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            return_all_layers=False,
            num_layers=num_layers,
        )
        self.classification_layer = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=num_classes,
            kernel_size=kernel_size,
            padding=1,
        )
        self.pad_value = pad_value


class ConvLSTMLightning(BaseModel):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        img_height_width: Tuple[int, int],
        kernel_size: Tuple[int, int],
        num_layers: int,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            required_img_size=img_height_width, # Important: ConvLSTM requires a fixed input size.
            *args,
            **kwargs
        )

        self.model = ConvLSTM_Seg_multi_layers(
            num_classes=1,
            input_size=img_height_width,
            input_dim=n_channels,
            hidden_dim=64,
            kernel_size=kernel_size,
            num_layers=num_layers,
        )
