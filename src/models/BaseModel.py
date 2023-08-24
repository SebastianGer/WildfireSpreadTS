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
    """_summary_ Base model class for all models in this project. Implements the training, validation and test steps, 
    as well as the loss function. 

    """
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
            n_channels (int): _description_ Number of feature channels in the input data. Usually means number of features per time step, 
            except for U-Net which flattens the temporal dimension and uses this parameter as the total number of features. 
            flatten_temporal_dimension (bool): _description_ Whether to flatten the temporal dimension of the input data.
            pos_class_weight (float): _description_ Weight of the positive class in the loss function (only used for BCE and Focal loss).
            loss_function (Literal[&#39;BCE&#39;, &#39;Focal&#39;, &#39;Lovasz&#39;, &#39;Jaccard&#39;, &#39;Dice&#39;]): _description_ Which loss function to use. 
            use_doy (bool, optional): _description_. Whether to use the doy of year (doy) as an additional input feature. Defaults to False.
            required_img_size (Optional[Tuple[int,int]], optional): _description_. Defaults to None. 
            When using a model that requires a specific image size, this parameter can be used to indicate it. We assume models require square images, 
            so this parameter indicates the side length. If set, the forward method will perform repeated inference on crops of the 
            image, and aggregate the results. This also works for non-square images. 
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

        # Plot PR curve at the end of training. Use fixed number of threshold to avoid the plot becoming 800MB+. 
        self.test_pr_curve = torchmetrics.PrecisionRecallCurve("binary", thresholds=100)

    def forward(self, x, doys=None):
        # If doys are used, the model needs to re-implement the forward method
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        return self.model(x)

    def get_pred_and_gt(self, batch):
        """_summary_ Unbatch the data and perform inference on each sample.

        Args:
            batch (_type_): _description_ Either a tuple of (x, y) or (x, y, doys).

        Raises:
            ValueError: _description_ If the batch size is not 1 and the model requires repeated inference on crops of the image. 
            This is the case for ConvLSTM, when predicting on the test set. During training, it uses random crops of the required size,
            so larger batch sizes can be used. 

        Returns:
            _type_: _description_ Prediction and ground truth for each sample in the batch.
        """

        # UTAE and TSViT use an additional doy feature as input. 
        if self.hparams.use_doy:
            x, y, doys = batch
        else:
            x, y = batch
            doys = None

        # If the model requires a certain fixed size, perform repeated inference on crops of the image,
        # and aggregate the results. When we reach the last row or column, which might not be divisible by
        # the required size, we align the crop window with the right/bottom edge of the image. This means 
        # that there is some amount of overlap between the last two crops in each row/column. We handle this
        # by simply overwriting the existing predictions with the new ones. 

        if self.hparams.required_img_size is not None:
            B, T, C, H, W = x.shape

            if x.shape[-2:] != self.hparams.required_img_size:
                if B != 1:
                    raise ValueError(
                        "Not implemented: repeated cropping for batch size > 1."
                    )
                # Use crops of size H_rq x W_rq
                H_req, W_req = self.hparams.required_img_size

                n_H = math.ceil(H / H_req)
                n_W = math.ceil(W / W_req)

                # Aggregate predictions in this tensor
                agg_output = torch.zeros(B, H, W, device=self.device)

                for i in range(n_H):
                    for j in range(n_W):
                        
                        # If we reach the bottom edge of the image, align the crop window with the bottom edge of the image
                        if i == n_H - 1:
                            H1 = H - H_req
                            H2 = H
                        else:
                            H1 = i * H_req
                            H2 = (i + 1) * H_req
                        # If we reach the right edge of the image, align the crop window with the right edge of the image
                        if j == n_W - 1:
                            W1 = W - W_req
                            W2 = W
                        else:
                            W1 = j * W_req
                            W2 = (j + 1) * W_req

                        x_crop = x[:, :, :, H1:H2, W1:W2]

                        agg_output[:, H1:H2, W1:W2] = self(x_crop, doys).squeeze(1)

                y_hat = agg_output
                return y_hat, y

        y_hat = self(x, doys).squeeze(1)

        return y_hat, y

    def training_step(self, batch, batch_idx):
        """_summary_ Compute predictions and loss for the given batch. Log training loss and F1 score.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
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
        """_summary_ Compute predictions and loss for the given batch. Log validation loss and F1 score.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
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
        )  
        return loss

    def test_step(self, batch, batch_idx):
        """_summary_ Compute predictions and loss for the given batch. Log test loss, F1, AP, precision, recall, IoU and confusion matrix.

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_hat, y = self.get_pred_and_gt(batch)

        loss = self.compute_loss(y_hat, y)
        self.test_f1(y_hat, y)
        self.test_avg_precision(y_hat, y)
        self.test_precision(y_hat, y)
        self.test_recall(y_hat, y)
        self.test_iou(y_hat, y)
        self.test_pr_curve.update(y_hat, y)
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
        """_summary_ Log the test PR curve and confusion matrix after predicting all test samples.
        """
        conf_mat = self.conf_mat.compute().cpu().numpy()
        wandb_table = wandb.Table(
            data=conf_mat, columns=["PredictedBackground", "PredictedFire"]
        )
        wandb.log({"Test confusion matrix": wandb_table})

        fig, ax = self.test_pr_curve.plot(score=True)
        wandb.log({"Test PR Curve": fig})

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
        else:
            return self.loss(y_hat, y.float())
