import torch
import julius
import torchopenl3
import torchmetrics
import pytorch_lightning as pl
from typing import Tuple, List, Dict
from argparse import ArgumentParser

from deepafx_st.probes.cdpam_encoder import CDPAMEncoder
from deepafx_st.probes.random_mel import RandomMelProjection

import deepafx_st.utils as utils
from deepafx_st.utils import DSPMode
from deepafx_st.system import System
from deepafx_st.data.style import StyleDataset


class ProbeSystem(pl.LightningModule):
    def __init__(
        self,
        audio_dir=None,
        num_classes=5,
        task="style",
        encoder_type="deepafx_st_autodiff",
        deepafx_st_autodiff_ckpt=None,
        deepafx_st_spsa_ckpt=None,
        deepafx_st_proxy0_ckpt=None,
        probe_type="linear",
        batch_size=32,
        lr=3e-4,
        lr_patience=20,
        patience=10,
        preload=False,
        sample_rate=24000,
        shuffle=True,
        num_workers=16,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if "deepafx_st" in self.hparams.encoder_type:

            if "autodiff" in self.hparams.encoder_type:
                self.hparams.deepafx_st_ckpt = self.hparams.deepafx_st_autodiff_ckpt
            elif "spsa" in self.hparams.encoder_type:
                self.hparams.deepafx_st_ckpt = self.hparams.deepafx_st_spsa_ckpt
            elif "proxy0" in self.hparams.encoder_type:
                self.hparams.deepafx_st_ckpt = self.hparams.deepafx_st_proxy0_ckpt

            else:
                raise RuntimeError(f"Invalid encoder_type: {self.hparams.encoder_type}")

            if self.hparams.deepafx_st_ckpt is None:
                raise RuntimeError(
                    f"Must supply {self.hparams.encoder_type}_ckpt checkpoint."
                )
            use_dsp = DSPMode.NONE
            system = System.load_from_checkpoint(
                self.hparams.deepafx_st_ckpt,
                use_dsp=use_dsp,
                batch_size=self.hparams.batch_size,
                spsa_parallel=False,
                proxy_ckpts=[],
                strict=False,
            )
            system.eval()
            self.encoder = system.encoder
            self.hparams.embed_dim = self.encoder.embed_dim

            # freeze weights
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

        elif self.hparams.encoder_type == "openl3":
            self.encoder = torchopenl3.models.load_audio_embedding_model(
                input_repr=self.hparams.openl3_input_repr,
                embedding_size=self.hparams.openl3_embedding_size,
                content_type=self.hparams.openl3_content_type,
            )
            self.hparams.embed_dim = 6144
        elif self.hparams.encoder_type == "random_mel":
            self.encoder = RandomMelProjection(
                self.hparams.sample_rate,
                self.hparams.random_mel_embedding_size,
                self.hparams.random_mel_n_mels,
                self.hparams.random_mel_n_fft,
                self.hparams.random_mel_hop_size,
            )
            self.hparams.embed_dim = self.hparams.random_mel_embedding_size
        elif self.hparams.encoder_type == "cdpam":
            self.encoder = CDPAMEncoder(self.hparams.cdpam_ckpt)
            self.encoder.eval()
            self.hparams.embed_dim = self.encoder.embed_dim
        else:
            raise ValueError(f"Invalid encoder_type: {self.hparams.encoder_type}")

        if self.hparams.probe_type == "linear":
            if self.hparams.task == "style":
                self.probe = torch.nn.Sequential(
                    torch.nn.Linear(self.hparams.embed_dim, self.hparams.num_classes),
                    # torch.nn.Softmax(-1),
                )
        elif self.hparams.probe_type == "mlp":
            if self.hparams.task == "style":
                self.probe = torch.nn.Sequential(
                    torch.nn.Linear(self.hparams.embed_dim, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, 512),
                    torch.nn.ReLU(),
                    torch.nn.Linear(512, self.hparams.num_classes),
                )
        self.accuracy = torchmetrics.Accuracy()
        self.f1_score = torchmetrics.F1Score(self.hparams.num_classes)

    def forward(self, x):
        bs, chs, samp = x.size()
        with torch.no_grad():
            if "deepafx_st" in self.hparams.encoder_type:
                x /= x.abs().max()
                x *= 10 ** (-12.0 / 20)  # with min 12 dBFS headroom
                e = self.encoder(x)
                norm = torch.norm(e, p=2, dim=-1, keepdim=True)
                e = e / norm
            elif self.hparams.encoder_type == "openl3":
                # x = julius.resample_frac(x, self.hparams.sample_rate, 48000)
                e, ts = torchopenl3.get_audio_embedding(
                    x,
                    48000,
                    model=self.encoder,
                    input_repr="mel128",
                    content_type="music",
                )
                e = e.permute(0, 2, 1)
                e = e.mean(dim=-1)
                # normalize by L2 norm
                norm = torch.norm(e, p=2, dim=-1, keepdim=True)
                e = e / norm
            elif self.hparams.encoder_type == "random_mel":
                e = self.encoder(x)
                norm = torch.norm(e, p=2, dim=-1, keepdim=True)
                e = e / norm
            elif self.hparams.encoder_type == "cdpam":
                # x = julius.resample_frac(x, self.hparams.sample_rate, 22050)
                x = torch.round(x * 32768)
                e = self.encoder(x)

        return self.probe(e)

    def common_step(
        self,
        batch: Tuple,
        batch_idx: int,
        optimizer_idx: int = 0,
        train: bool = True,
    ):
        loss = 0
        x, y = batch

        y_hat = self(x)

        # compute CE
        if self.hparams.task == "style":
            loss = torch.nn.functional.cross_entropy(y_hat, y)

        if not train:
            # store audio data
            data_dict = {"x": x.float().cpu()}
        else:
            data_dict = {}

        self.log(
            "train_loss" if train else "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        if not train and self.hparams.task == "style":
            self.log("val_acc_step", self.accuracy(y_hat, y))
            self.log("val_f1_step", self.f1_score(y_hat, y))

        return loss, data_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, _ = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, data_dict = self.common_step(batch, batch_idx, train=False)

        if batch_idx == 0:
            return data_dict

    def validation_epoch_end(self, outputs) -> None:
        if self.hparams.task == "style":
            self.log("val_acc_epoch", self.accuracy.compute())
            self.log("val_f1_epoch", self.f1_score.compute())

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.probe.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
        )

        ms1 = int(self.hparams.max_epochs * 0.8)
        ms2 = int(self.hparams.max_epochs * 0.95)
        print(
            "Learning rate schedule:",
            f"0 {self.hparams.lr:0.2e} -> ",
            f"{ms1} {self.hparams.lr*0.1:0.2e} -> ",
            f"{ms2} {self.hparams.lr*0.01:0.2e}",
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[ms1, ms2],
            gamma=0.1,
        )

        return [optimizer], {"scheduler": scheduler, "monitor": "val_loss"}

    def train_dataloader(self):

        if self.hparams.task == "style":
            train_dataset = StyleDataset(
                self.hparams.audio_dir,
                "train",
                sample_rate=self.hparams.encoder_sample_rate,
            )

        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(
            train_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            worker_init_fn=utils.seed_worker,
            generator=g,
            pin_memory=True,
        )

    def val_dataloader(self):

        if self.hparams.task == "style":
            val_dataset = StyleDataset(
                self.hparams.audio_dir,
                subset="val",
                sample_rate=self.hparams.encoder_sample_rate,
            )

        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(
            val_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            worker_init_fn=utils.seed_worker,
            generator=g,
            pin_memory=True,
        )

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- Model  ---
        parser.add_argument("--encoder_type", type=str, default="deeapfx2")
        parser.add_argument("--probe_type", type=str, default="linear")
        parser.add_argument("--task", type=str, default="style")
        parser.add_argument("--encoder_sample_rate", type=int, default=24000)
        # --- deeapfx2  ---
        parser.add_argument("--deepafx_st_autodiff_ckpt", type=str)
        parser.add_argument("--deepafx_st_spsa_ckpt", type=str)
        parser.add_argument("--deepafx_st_proxy0_ckpt", type=str)

        # --- cdpam  ---
        parser.add_argument("--cdpam_ckpt", type=str)
        # --- openl3  ---
        parser.add_argument("--openl3_input_repr", type=str, default="mel128")
        parser.add_argument("--openl3_content_type", type=str, default="env")
        parser.add_argument("--openl3_embedding_size", type=int, default=6144)
        # --- random_mel  ---
        parser.add_argument("--random_mel_embedding_size", type=str, default=4096)
        parser.add_argument("--random_mel_n_fft", type=str, default=4096)
        parser.add_argument("--random_mel_hop_size", type=str, default=1024)
        parser.add_argument("--random_mel_n_mels", type=str, default=128)
        # --- Training  ---
        parser.add_argument("--audio_dir", type=str)
        parser.add_argument("--num_classes", type=int, default=5)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_patience", type=int, default=20)
        parser.add_argument("--patience", type=int, default=10)
        parser.add_argument("--preload", action="store_true")
        parser.add_argument("--sample_rate", type=int, default=24000)
        parser.add_argument("--num_workers", type=int, default=8)

        return parser
