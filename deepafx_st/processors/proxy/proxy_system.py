from re import X
import torch
import auraloss
import pytorch_lightning as pl
from typing import Tuple, List, Dict
from argparse import ArgumentParser


import deepafx_st.utils as utils
from deepafx_st.data.proxy import DSPProxyDataset
from deepafx_st.processors.proxy.tcn import ConditionalTCN
from deepafx_st.processors.spsa.channel import SPSAChannel
from deepafx_st.processors.dsp.peq import ParametricEQ
from deepafx_st.processors.dsp.compressor import Compressor


class ProxySystem(pl.LightningModule):
    def __init__(
        self,
        causal=True,
        nblocks=4,
        dilation_growth=8,
        kernel_size=13,
        channel_width=64,
        input_dir=None,
        processor="channel",
        batch_size=32,
        lr=3e-4,
        lr_patience=20,
        patience=10,
        preload=False,
        sample_rate=24000,
        shuffle=True,
        train_length=65536,
        train_examples_per_epoch=10000,
        val_length=131072,
        val_examples_per_epoch=1000,
        num_workers=16,
        output_gain=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        #print(f"Proxy Processor: {processor} @ fs={sample_rate} Hz")

        # construct both the true DSP...
        if self.hparams.processor == "peq":
            self.processor = ParametricEQ(self.hparams.sample_rate)
        elif self.hparams.processor == "comp":
            self.processor = Compressor(self.hparams.sample_rate)
        elif self.hparams.processor == "channel":
            self.processor = SPSAChannel(self.hparams.sample_rate)

        # and the neural network proxy
        self.proxy = ConditionalTCN(
            self.hparams.sample_rate,
            num_control_params=self.processor.num_control_params,
            causal=self.hparams.causal,
            nblocks=self.hparams.nblocks,
            channel_width=self.hparams.channel_width,
            kernel_size=self.hparams.kernel_size,
            dilation_growth=self.hparams.dilation_growth,
        )

        self.receptive_field = self.proxy.compute_receptive_field()

        self.recon_losses = {}
        self.recon_loss_weights = {}

        self.recon_losses["mrstft"] = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[32, 128, 512, 2048, 8192, 32768],
            hop_sizes=[16, 64, 256, 1024, 4096, 16384],
            win_lengths=[32, 128, 512, 2048, 8192, 32768],
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        )
        self.recon_loss_weights["mrstft"] = 1.0

        self.recon_losses["l1"] = torch.nn.L1Loss()
        self.recon_loss_weights["l1"] = 100.0

    def forward(self, x, p, use_dsp=False, sample_rate=24000, **kwargs):
        """Use the pre-trained neural network proxy effect."""
        bs, chs, samp = x.size()
        if not use_dsp:
            y = self.proxy(x, p)
            # manually apply the makeup gain parameter
            if self.hparams.output_gain and not self.hparams.processor == "peq":
                gain_db = (p[..., -1] * 96) - 48
                gain_ln = 10 ** (gain_db / 20.0)
                y *= gain_ln.view(bs, chs, 1)
        else:
            with torch.no_grad():
                bs, chs, s = x.shape

                if self.hparams.output_gain and not self.hparams.processor == "peq":
                    # override makeup gain
                    gain_db = (p[..., -1] * 96) - 48
                    gain_ln = 10 ** (gain_db / 20.0)
                    p[..., -1] = 0.5

                if self.hparams.processor == "channel":
                    y_temp = self.processor(x.cpu(), p.cpu())
                    y_temp = y_temp.view(bs, chs, s).type_as(x)
                else:
                    y_temp = self.processor(
                        x.cpu().numpy(),
                        p.cpu().numpy(),
                        sample_rate,
                    )
                    y_temp = torch.tensor(y_temp).view(bs, chs, s).type_as(x)

                y = y_temp.type_as(x).view(bs, 1, -1)

                if self.hparams.output_gain and not self.hparams.processor == "peq":
                    y *= gain_ln.view(bs, chs, 1)

        return y

    def common_step(
        self,
        batch: Tuple,
        batch_idx: int,
        optimizer_idx: int = 0,
        train: bool = True,
    ):
        loss = 0
        x, y, p = batch

        y_hat = self(x, p)

        # compute loss
        for loss_idx, (loss_name, loss_fn) in enumerate(self.recon_losses.items()):
            tmp_loss = loss_fn(y_hat.float(), y.float())
            loss += self.recon_loss_weights[loss_name] * tmp_loss

            self.log(
                f"train_loss/{loss_name}" if train else f"val_loss/{loss_name}",
                tmp_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        if not train:
            # store audio data
            data_dict = {
                "x": x.float().cpu(),
                "y": y.float().cpu(),
                "p": p.float().cpu(),
                "y_hat": y_hat.float().cpu(),
            }
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

        return loss, data_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, _ = self.common_step(batch, batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, data_dict = self.common_step(batch, batch_idx, train=False)

        if batch_idx == 0:
            return data_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.proxy.parameters(),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.hparams.lr_patience,
            verbose=True,
        )

        return [optimizer], {"scheduler": scheduler, "monitor": "val_loss"}

    def train_dataloader(self):

        train_dataset = DSPProxyDataset(
            self.hparams.input_dir,
            self.processor,
            self.hparams.processor,  # name
            subset="train",
            length=self.hparams.train_length,
            num_examples_per_epoch=self.hparams.train_examples_per_epoch,
            half=True if self.hparams.precision == 16 else False,
            buffer_size_gb=self.hparams.buffer_size_gb,
            buffer_reload_rate=self.hparams.buffer_reload_rate,
        )

        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(
            train_dataset,
            num_workers=self.hparams.num_workers,
            batch_size=self.hparams.batch_size,
            worker_init_fn=utils.seed_worker,
            generator=g,
            pin_memory=True,
        )

    def val_dataloader(self):

        val_dataset = DSPProxyDataset(
            self.hparams.input_dir,
            self.processor,
            self.hparams.processor,  # name
            subset="val",
            length=self.hparams.val_length,
            num_examples_per_epoch=self.hparams.val_examples_per_epoch,
            half=True if self.hparams.precision == 16 else False,
            buffer_size_gb=self.hparams.buffer_size_gb,
            buffer_reload_rate=self.hparams.buffer_reload_rate,
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

    @staticmethod
    def count_control_params(plugin_config):
        num_control_params = 0

        for plugin in plugin_config["plugins"]:
            for port in plugin["ports"]:
                if port["optim"]:
                    num_control_params += 1

        return num_control_params

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- Model  ---
        parser.add_argument("--causal", action="store_true")
        parser.add_argument("--output_gain", action="store_true")
        parser.add_argument("--dilation_growth", type=int, default=8)
        parser.add_argument("--nblocks", type=int, default=4)
        parser.add_argument("--kernel_size", type=int, default=13)
        parser.add_argument("--channel_width", type=int, default=13)
        # --- Training  ---
        parser.add_argument("--input_dir", type=str)
        parser.add_argument("--processor", type=str)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_patience", type=int, default=20)
        parser.add_argument("--patience", type=int, default=10)
        parser.add_argument("--preload", action="store_true")
        parser.add_argument("--sample_rate", type=int, default=24000)
        parser.add_argument("--shuffle", type=bool, default=True)
        parser.add_argument("--train_length", type=int, default=65536)
        parser.add_argument("--train_examples_per_epoch", type=int, default=10000)
        parser.add_argument("--val_length", type=int, default=131072)
        parser.add_argument("--val_examples_per_epoch", type=int, default=1000)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--buffer_reload_rate", type=int, default=1000)
        parser.add_argument("--buffer_size_gb", type=float, default=1.0)

        return parser
