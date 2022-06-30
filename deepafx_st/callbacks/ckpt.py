import os
import sys
import shutil
import pytorch_lightning as pl


class CopyPretrainedCheckpoints(pl.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer, pl_module):
        """Before training, move the pre-trained checkpoints
        to the current checkpoint directory.

        """
        # copy any pre-trained checkpoints to new directory
        if pl_module.hparams.processor_model == "proxy":
            pretrained_ckpt_dir = os.path.join(
                pl_module.logger.experiment.log_dir, "pretrained_checkpoints"
            )
            if not os.path.isdir(pretrained_ckpt_dir):
                os.makedirs(pretrained_ckpt_dir)
            cp_proxy_ckpts = []
            for proxy_ckpt in pl_module.hparams.proxy_ckpts:
                new_ckpt = shutil.copy(
                    proxy_ckpt,
                    pretrained_ckpt_dir,
                )
                cp_proxy_ckpts.append(new_ckpt)
                print(f"Moved checkpoint to {new_ckpt}.")
            # overwrite to the paths in current experiment logs
            pl_module.hparams.proxy_ckpts = cp_proxy_ckpts
            print(pl_module.hparams.proxy_ckpts)
