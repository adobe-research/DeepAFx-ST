import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from deepafx_st.probes.probe_system import ProbeSystem

torch.backends.cudnn.benchmark = True
pl.seed_everything(42)

# some arg parse for configuration
parser = ArgumentParser()

# add all the available trainer and system options to argparse
parser = pl.Trainer.add_argparse_args(parser)
parser = ProbeSystem.add_model_specific_args(parser)

# parse them args
args = parser.parse_args()

# setup callbacks
callbacks = [
    pl.callbacks.ModelCheckpoint(
        monitor="val_f1_epoch",
        mode="max",
        filename="{epoch}-{step}-val-" + f"{args.encoder_type}-{args.probe_type}",
    ),
]

# create PyTorch Lightning trainer
trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
system = ProbeSystem(**vars(args))

# train!
trainer.fit(system)
