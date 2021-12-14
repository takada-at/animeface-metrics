from argparse import ArgumentParser
from pathlib import Path
from animeface_metric import model, data

import pytorch_lightning as pl
import torch


def _train_inner(args, train_dataloader, valid_dataloader):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    default_opts = dict(
        checkpoint_callback=checkpoint_callback,
    )
    params = vars(args)
    params["per_epoch"] = len(train_dataloader)
    if args.max_steps:
        params["max_steps"] = args.max_steps
    else:
        params["max_steps"] = args.max_epochs * params["per_epoch"]
    if args.checkpoint:
        md = model.MetricsLearningModel.load_from_checkpoint(str(args.checkpoint))
        trainer = pl.Trainer.from_argparse_args(
            args, resume_from_checkpoint=str(args.checkpoint), **default_opts
        )
    else:
        md = model.MetricsLearningModel(**params)
        trainer = pl.Trainer.from_argparse_args(args, **default_opts)
    if args.auto_lr_find:
        lr_finder = trainer.tuner.lr_find(md, train_dataloader, valid_dataloader)
        print(lr_finder.results)
        new_lr = lr_finder.suggestion()
        print(f"lr={new_lr}")
        md.hparams.lr = new_lr
    trainer.fit(md, train_dataloader, valid_dataloader)
    print(trainer.checkpoint_callback.best_model_score)
    return float(trainer.checkpoint_callback.best_model_score)


def train(args):
    train_dataloader, valid_dataloader = data.get_data_loader(args)
    score = _train_inner(args, train_dataloader, valid_dataloader)
    print(f"score={score}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--disable_cudnn", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser = pl.Trainer.add_argparse_args(parser)
    script_args, _ = parser.parse_known_args()
    parser = model.MetricsLearningModel.add_model_specific_args(parser)
    parser = data.add_data_args(parser)
    args = parser.parse_args()
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    train(args)


if __name__ == "__main__":
    main()
