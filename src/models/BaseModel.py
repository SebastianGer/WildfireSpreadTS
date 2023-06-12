import math
from abc import ABC
from typing import Any, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import wandb
from segmentation_models_pytorch.losses import (DiceLoss, JaccardLoss,
                                                LovaszLoss)
from torchvision.ops import sigmoid_focal_loss


class BaseModel(pl.LightningModule, ABC):
    def __init__(
        self,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        loss_function: Literal["BCE", "Focal", "Lovasz", "Jaccard", "Dice"],
        use_doy: bool = False,
        required_img_size: Optional[Tuple[int, int]] = None,
        *args: Any,
        **kwargs: Any
    ):
        """_summary_

        Args:
            n_channels (int): _description_
            flatten_temporal_dimension (bool): _description_
            pos_class_weight (float): _description_
            loss_function (Literal[&#39;BCE&#39;, &#39;Focal&#39;, &#39;Lovasz&#39;, &#39;Jaccard&#39;, &#39;Dice&#39;]): _description_
            use_doy (bool, optional): _description_. Defaults to False.
            required_img_size (Optional[Tuple[int,int]], optional): _description_. Defaults to None. When using a model that requires a specific image size, this parameter can be used to indicate it, and react accordingly in the forward method. We assume square images, so this parameter indicates the side length.
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        if required_img_size is not None:
            self.hparams.required_img_size = torch.Size(
                required_img_size, device=self.device
            )

        # Normalize class weights by assuming that the negative class has weight 1
        if self.hparams.loss_function == "Focal" and self.hparams.pos_class_weight > 1:
            self.hparams.pos_class_weight /= 1 + self.hparams.pos_class_weight

        self.loss = self.get_loss()

        self.train_f1 = torchmetrics.F1Score("binary")
        self.val_f1 = self.train_f1.clone()
        self.test_f1 = self.train_f1.clone()

        self.test_avg_precision = torchmetrics.AveragePrecision("binary")
        self.test_precision = torchmetrics.Precision("binary")
        self.test_recall = torchmetrics.Recall("binary")
        self.test_iou = torchmetrics.JaccardIndex("binary")
        self.conf_mat = torchmetrics.ConfusionMatrix("binary")

    def forward(self, x, doys=None):
        # If doys are used, the model needs to re-implement the forward method
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        return self.model(x)

    def get_pred_and_gt(self, batch):
        # UTAE and TSViT use an additional doy feature as input. By implementing this method here,
        # we can reuse the train/val/test_step methods and only the forward method needs to be re-implemented
        # by the respective model.
        if self.hparams.use_doy:
            x, y, doys = batch
        else:
            x, y = batch
            doys = None

        # If the model requires a certain fixed size, perform repeated inference on crops of the image,
        # and aggregate the results.
        # We use replication padding to make the image size divisible by the required size, padding on all sides equally.
        # We use replication padding, to avoid creating values that would suggest wrong conditions (e.g. mean value,
        # when the neighboring features are not close to the mean), and to avoid reflection padding, which would
        # require handling features like wind direction differently, since they indicate a direction on the grid.

        if self.hparams.required_img_size is not None:
            B, T, C, H, W = x.shape

            if x.shape[-2:] != self.hparams.required_img_size:
                if B != 1:
                    raise ValueError(
                        "Not implemented: repeated cropping for batch size > 1. This is a limitation of the"
                        + "padding function we use. "
                    )
                H_req, W_req = self.hparams.required_img_size

                H_padded = H_req * math.ceil(H / H_req)
                W_padded = W_req * math.ceil(W / W_req)

                W_padding = W_padded - W
                W_padding_left = W_padding // 2
                W_padding_right = W_padding - W_padding_left
                H_padding = H_padded - H
                H_padding_top = H_padding // 2
                H_padding_bottom = H_padding - H_padding_top

                # Padding is limited in terms of number of dimensions, so we can only pad batches of size 1.
                # Since we only want to use this for the testing phase, this is fine for now.
                x_padded = torch.nn.functional.pad(
                    x[0, ...],
                    (W_padding_left, W_padding_right,
                     H_padding_top, H_padding_bottom),
                    mode="replicate",
                ).unsqueeze(0)

                agg_output = torch.zeros(
                    B, H_padded, W_padded, device=self.device)

                for i in range(H // H_req):
                    for j in range(W // W_req):
                        x_crop = x_padded[
                            :,
                            :,
                            :,
                            i * H_req: (i + 1) * H_req,
                            j * W_req: (j + 1) * W_req,
                        ]
                        agg_output[
                            :, i * H_req: (i + 1) * H_req, j * W_req: (j + 1) * W_req
                        ] = self(x_crop, doys).squeeze(1)

                y_hat = agg_output[
                    :,
                    (H_padding_top): (H + H_padding_top),
                    (W_padding_left): (W + W_padding_left),
                ]
                return y_hat, y

        y_hat = self(x, doys).squeeze(1)

        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)

        loss = self.compute_loss(y_hat, y)
        f1 = self.train_f1(y_hat, y)
        self.log(
            "train_loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)

        loss = self.compute_loss(y_hat, y)
        f1 = self.val_f1(y_hat, y)
        self.log(
            "val_loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )  # no sync_dist, because theoretically this should sync automatically as a torchmetrics object
        return loss

    def test_step(self, batch, batch_idx):
        y_hat, y = self.get_pred_and_gt(batch)

        loss = self.compute_loss(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_avg_precision(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_iou(y_hat, y)
        self.conf_mat.update(y_hat, y)

        self.log("test_loss", loss.item(), sync_dist=True)
        self.log_dict(
            {
                "test_f1": self.test_f1,
                "test_AP": self.test_avg_precision,
                "test_precision": self.test_precision,
                "test_recall": self.test_recall,
                "test_iou": self.test_iou,
            }
        )
        return loss

    def on_test_epoch_end(self) -> None:
        conf_mat = self.conf_mat.compute().cpu().numpy()
        wandb_table = wandb.Table(
            data=conf_mat, columns=["PredictedBackground", "PredictedFire"]
        )
        wandb.log({"Test confusion matrix": wandb_table})

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        x_af = x[:, :, -1, :, :]
        y_hat = self(x).squeeze(1)
        return x_af, y, y_hat

    def get_loss(self):
        if self.hparams.loss_function == "BCE":
            return nn.BCEWithLogitsLoss(
                pos_weight=torch.Tensor(
                    [self.hparams.pos_class_weight], device=self.device
                )
            )
        elif self.hparams.loss_function == "Focal":
            return sigmoid_focal_loss
        elif self.hparams.loss_function == "Lovasz":
            return LovaszLoss(mode="binary")
        elif self.hparams.loss_function == "Jaccard":
            return JaccardLoss(mode="binary")
        elif self.hparams.loss_function == "Dice":
            return DiceLoss(mode="binary")

    def compute_loss(self, y_hat, y):
        if self.hparams.loss_function == "Focal":
            return self.loss(
                y_hat,
                y.float(),
                alpha=1 - self.hparams.pos_class_weight,
                gamma=2,
                reduction="mean",
            )
        elif self.hparams.loss_function == "BCE":
            return self.loss(y_hat, y.float())
        # segmentation models pytorch losses
        else:
            return self.loss(y_hat, y.float())
