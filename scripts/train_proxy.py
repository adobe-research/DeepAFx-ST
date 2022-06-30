import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser

from deepafx_st.processors.proxy.proxy_system import ProxySystem
from deepafx_st.callbacks.audio import LogAudioCallback

torch.backends.cudnn.benchmark = True
pl.seed_everything(42)

# some arg parse for configuration
parser = ArgumentParser()

# add all the available trainer and system options to argparse
parser = pl.Trainer.add_argparse_args(parser)
parser = ProxySystem.add_model_specific_args(parser)

# parse them args
args = parser.parse_args()

dataset_name = args.default_root_dir.split(os.sep)[-2]

# setup callbacks
callbacks = [
    LogAudioCallback(),
    pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch}-{step}-val-" + f"{dataset_name}-{args.processor}",
    ),
]

# create PyTorch Lightning trainer
trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
system = ProxySystem(**vars(args))

# train!
trainer.fit(system)
