import torch

from animeface_metric import metric_learning
from argparse import ArgumentParser
from torch import optim
from torch.optim import lr_scheduler

import enum
import numpy as np
import pytorch_lightning as pl
import timm
import torch.nn as nn
import torchmetrics


class LossModule(enum.Enum):
    arcface = "arcface"
    cosface = "cosface"
    adacos = "adacos"


def get_loss_module(
    module_type: LossModule,
    in_features: int,
    n_classes: int,
    s: float = 30.0,
    margin: float = 0.50,
):
    if module_type is LossModule.arcface:
        module = metric_learning.ArcFace(
            in_features,
            n_classes,
            s=s,
            m=margin,
        )
    elif module_type is LossModule.cosface:
        module = metric_learning.CosFace(in_features, n_classes, s=s, m=margin)
    elif module_type is LossModule.adacos:
        module = metric_learning.AdaCos(
            in_features, n_classes, m=margin,
        )
    else:
        raise ValueError
    return module


def get_lr_scheduler(optimizer, max_epochs):
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6, last_epoch=-1
    )
    scheduler = {
        "scheduler": scheduler,
        "interval": "epoch",
        "frequency": 1,
    }
    return scheduler


class MetricsLearningModel(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--backbone_name", type=str, default="resnet34")
        parser.add_argument("--warmup_steps", type=float, default=0)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--weight_norm", action="store_true")
        parser.add_argument("--pretrained", action="store_true")
        parser.add_argument("--use_fc", action="store_true")
        parser.add_argument(
            "--loss_module", type=LossModule, default=LossModule.arcface
        )
        parser.add_argument("--scale", type=float, default=30.0)
        parser.add_argument("--margin", type=float, default=0.50)
        parser.add_argument("--disable_fc", action="store_true")

        return parser

    def __init__(
        self,
        learning_rate: float,
        backbone_name: str,
        pretrained: bool = False,
        per_epoch: int = 128,
        max_steps: int = 2500,
        warmup_steps: int = 0,
        weight_decay: float = 0,
        loss_module: LossModule = LossModule.arcface,
        n_classes: int = 1000,
        scale: float = 30.0,
        margin: float = 0.50,
        dropout: float = 0.1,
        fc_dim: int = 512,
        disable_fc: bool = False,
        **kwargs,
    ):
        super(MetricsLearningModel, self).__init__()
        model_parameters = dict(pretrained=pretrained, in_chans=3)
        backbone = timm.create_model(backbone_name, **model_parameters)
        classifier = backbone.get_classifier()
        in_features = classifier.in_features
        backbone.reset_classifier(0, "")
        self.backbone = backbone
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.loss_module = get_loss_module(
            module_type=loss_module,
            in_features=in_features,
            n_classes=n_classes,
            s=scale,
            margin=margin,
        )
        if disable_fc:
            self.classifier = None
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, fc_dim),
                nn.BatchNorm1d(fc_dim),
            )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.parameters(), learning_rate, weight_decay=weight_decay
        )
        self.max_epochs = max_steps // per_epoch
        self.lr_scheduler = get_lr_scheduler(self.optimizer, max_epochs=self.max_epochs)
        self.warmup_steps = warmup_steps * self.max_epochs
        self.acc = torchmetrics.Accuracy()
        self.f1 = torchmetrics.F1()
        self.save_hyperparameters()

    def predict(self, x):
        x = self.forward_features(x)
        return x

    def forward_features(self, x):
        x = self.backbone.forward_features(x)
        x = self.pooling(x).reshape(x.shape[0], x.shape[1])
        return x

    def forward(self, x, label=None):
        x = self.forward_features(x)
        if self.classifier is not None:
            x = self.classifier(x)
        logits = self.loss_module(x, label)
        return logits, x

    def calc_loss(self, batch):
        label = batch["label"]
        if self.training:
            logits, x = self(batch["image"], label)
        else:
            logits, x = self(batch["image"], None)
        loss = self.criterion(logits, label)
        return dict(loss=loss, logits=logits)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        results = self.calc_loss(batch)
        loss = results["loss"]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        results = self.calc_loss(batch)
        loss = results["loss"]
        label = batch["label"]
        return dict(val_loss=loss, label=label, logits=results["logits"])

    def validation_epoch_end(self, outputs) -> None:
        loss = np.mean([o["val_loss"].mean().item() for o in outputs])
        logits = torch.cat([o["logits"] for o in outputs], dim=0)
        label = torch.cat([o["label"] for o in outputs], dim=0)
        acc = self.acc(logits, label)
        f1 = self.f1(logits, label)
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        metrics = {"val_loss": loss, "lr": lr, "val_acc": acc, "val_f1": f1}
        self.log_dict(metrics, logger=True, prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        conf = {"optimizer": self.optimizer, "monitor": "val_loss"}
        if self.lr_scheduler is not None:
            conf["lr_scheduler"] = self.lr_scheduler
        return conf

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_idx: int = None,
        optimizer_closure=None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
        **kwargs,
    ) -> None:
        # warm up lr
        if self.warmup_steps > 0 and self.trainer.global_step < self.warmup_steps:
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / float(self.warmup_steps)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
