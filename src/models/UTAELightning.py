from typing import Any

import torch

from .BaseModel import BaseModel
from .utae_paps_models.utae import UTAE


class UTAELightning(BaseModel):
    """_summary_ U-Net architecture with temporal attention in the bottleneck and skip connections.
    """
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
            use_doy=True, # UTAE uses the day of the year as an input feature
            *args,
            **kwargs
        )

        self.model = UTAE(
            input_dim=n_channels,
            encoder_widths=[64, 64, 64, 128],
            decoder_widths=[32, 32, 64, 128],
            out_conv=[32, 1],
            str_conv_k=4,
            str_conv_s=2,
            str_conv_p=1,
            agg_mode="att_group",
            encoder_norm="group",
            n_head=16,
            d_model=256,
            d_k=4,
            encoder=False,
            return_maps=False,
            pad_value=0,
            padding_mode="reflect",
        )

    def forward(self, x: torch.Tensor, doys: torch.Tensor) -> torch.Tensor:
        return self.model(x, batch_positions=doys, return_att=False)
