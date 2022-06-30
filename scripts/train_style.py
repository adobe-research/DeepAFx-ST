import os
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.plugins import DDPPlugin

from deepafx_st.system import System
from deepafx_st.utils import system_summary
from deepafx_st.callbacks.audio import LogAudioCallback
from deepafx_st.callbacks.params import LogParametersCallback
from deepafx_st.callbacks.ckpt import CopyPretrainedCheckpoints

if __name__ == "__main__":

    torch.multiprocessing.set_start_method("spawn")

    torch.backends.cudnn.benchmark = True
    pl.seed_everything(42)

    # some arg parse for configuration
    parser = ArgumentParser()

    # add all the available trainer and system options to argparse
    parser = pl.Trainer.add_argparse_args(parser)
    parser = System.add_model_specific_args(parser)

    # parse them args
    args = parser.parse_args()

    # Checkpoint on the first reconstruction loss
    args.train_monitor = f"train_loss/{args.recon_losses[-1]}"
    args.val_monitor = f"val_loss/{args.recon_losses[-1]}"

    dataset_name = args.default_root_dir.split(os.sep)[-2]

    # setup callbacks
    callbacks = [
        LogAudioCallback(),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.ModelCheckpoint(
            monitor=args.train_monitor,
            filename="{epoch}-{step}-train-" + f"{dataset_name}-{args.processor_model}",
        ),
        pl.callbacks.ModelCheckpoint(
            monitor=args.val_monitor,
            filename="{epoch}-{step}-val-" + f"{dataset_name}-{args.processor_model}",
        ),
        CopyPretrainedCheckpoints(),
    ]

    if args.processor_model != "tcn":
        callbacks.append(LogParametersCallback())

    # create PyTorch Lightning trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        plugins=DDPPlugin(find_unused_parameters=False),
    )

    # create the System
    system = System(**vars(args))

    # print details about the model
    system_summary(system)

    # train!
    trainer.fit(system)

    # close threads
    del system.processor
