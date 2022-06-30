import torch
import auraloss
import torchaudio
from itertools import chain
import pytorch_lightning as pl
from argparse import ArgumentParser
from typing import Tuple, List, Dict

import deepafx_st.utils as utils
from deepafx_st.utils import DSPMode
from deepafx_st.data.dataset import AudioDataset
from deepafx_st.models.encoder import SpectralEncoder
from deepafx_st.models.controller import StyleTransferController
from deepafx_st.processors.spsa.channel import SPSAChannel
from deepafx_st.processors.spsa.eps_scheduler import EpsilonScheduler
from deepafx_st.processors.proxy.channel import ProxyChannel
from deepafx_st.processors.autodiff.channel import AutodiffChannel


class System(pl.LightningModule):
    def __init__(
        self,
        ext="wav",
        dsp_sample_rate=24000,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.eps_scheduler = EpsilonScheduler(
            self.hparams.spsa_epsilon,
            self.hparams.spsa_patience,
            self.hparams.spsa_factor,
            self.hparams.spsa_verbose,
        )

        self.hparams.dsp_mode = DSPMode.NONE

        # first construct the processor, since this will dictate encoder
        if self.hparams.processor_model == "spsa":
            self.processor = SPSAChannel(
                self.hparams.dsp_sample_rate,
                self.hparams.spsa_parallel,
                self.hparams.batch_size,
            )
        elif self.hparams.processor_model == "autodiff":
            self.processor = AutodiffChannel(self.hparams.dsp_sample_rate)
        elif self.hparams.processor_model == "proxy0":
            # print('self.hparams.proxy_ckpts,',self.hparams.proxy_ckpts)
            self.hparams.dsp_mode = DSPMode.NONE
            self.processor = ProxyChannel(
                self.hparams.proxy_ckpts,
                self.hparams.freeze_proxies,
                self.hparams.dsp_mode,
                sample_rate=self.hparams.dsp_sample_rate,
            )
        elif self.hparams.processor_model == "proxy1":
            # print('self.hparams.proxy_ckpts,',self.hparams.proxy_ckpts)
            self.hparams.dsp_mode = DSPMode.INFER
            self.processor = ProxyChannel(
                self.hparams.proxy_ckpts,
                self.hparams.freeze_proxies,
                self.hparams.dsp_mode,
                sample_rate=self.hparams.dsp_sample_rate,
            )
        elif self.hparams.processor_model == "proxy2":
            # print('self.hparams.proxy_ckpts,',self.hparams.proxy_ckpts)
            self.hparams.dsp_mode = DSPMode.TRAIN_INFER
            self.processor = ProxyChannel(
                self.hparams.proxy_ckpts,
                self.hparams.freeze_proxies,
                self.hparams.dsp_mode,
                sample_rate=self.hparams.dsp_sample_rate,
            )
        elif self.hparams.processor_model == "tcn1":
            # self.processor = ConditionalTCN(self.hparams.sample_rate)
            self.hparams.dsp_mode = DSPMode.NONE
            self.processor = ProxyChannel(
                [],
                freeze_proxies=False,
                dsp_mode=self.hparams.dsp_mode,
                tcn_nblocks=self.hparams.tcn_nblocks,
                tcn_dilation_growth=self.hparams.tcn_dilation_growth,
                tcn_channel_width=self.hparams.tcn_channel_width,
                tcn_kernel_size=self.hparams.tcn_kernel_size,
                num_tcns=1,
                sample_rate=self.hparams.sample_rate,
            )
        elif self.hparams.processor_model == "tcn2":
            self.hparams.dsp_mode = DSPMode.NONE
            self.processor = ProxyChannel(
                [],
                freeze_proxies=False,
                dsp_mode=self.hparams.dsp_mode,
                tcn_nblocks=self.hparams.tcn_nblocks,
                tcn_dilation_growth=self.hparams.tcn_dilation_growth,
                tcn_channel_width=self.hparams.tcn_channel_width,
                tcn_kernel_size=self.hparams.tcn_kernel_size,
                num_tcns=2,
                sample_rate=self.hparams.sample_rate,
            )
        else:
            raise ValueError(f"Invalid processor_model: {self.hparams.processor_model}")

        if self.hparams.encoder_ckpt is not None:
            # load encoder weights from a pre-trained system
            system = System.load_from_checkpoint(self.hparams.encoder_ckpt)
            self.encoder = system.encoder
            self.hparams.encoder_embed_dim = system.encoder.embed_dim
        else:
            self.encoder = SpectralEncoder(
                self.processor.num_control_params,
                self.hparams.sample_rate,
                encoder_model=self.hparams.encoder_model,
                embed_dim=self.hparams.encoder_embed_dim,
                width_mult=self.hparams.encoder_width_mult,
            )

        if self.hparams.encoder_freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.controller = StyleTransferController(
            self.processor.num_control_params,
            self.hparams.encoder_embed_dim,
        )

        if len(self.hparams.recon_losses) != len(self.hparams.recon_loss_weights):
            raise ValueError("Must supply same number of weights as losses.")

        self.recon_losses = torch.nn.ModuleDict()
        for recon_loss in self.hparams.recon_losses:
            if recon_loss == "mrstft":
                self.recon_losses[recon_loss] = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[32, 128, 512, 2048, 8192, 32768],
                    hop_sizes=[16, 64, 256, 1024, 4096, 16384],
                    win_lengths=[32, 128, 512, 2048, 8192, 32768],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            elif recon_loss == "mrstft-md":
                self.recon_losses[recon_loss] = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[128, 512, 2048, 8192],
                    hop_sizes=[32, 128, 512, 2048],  #  1 / 4
                    win_lengths=[128, 512, 2048, 8192],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            elif recon_loss == "mrstft-sm":
                self.recon_losses[recon_loss] = auraloss.freq.MultiResolutionSTFTLoss(
                    fft_sizes=[512, 2048, 8192],
                    hop_sizes=[256, 1024, 4096],  #  1 / 4
                    win_lengths=[512, 2048, 8192],
                    w_sc=0.0,
                    w_phs=0.0,
                    w_lin_mag=1.0,
                    w_log_mag=1.0,
                )
            elif recon_loss == "melfft":
                self.recon_losses[recon_loss] = auraloss.freq.MelSTFTLoss(
                    self.hparams.sample_rate,
                    fft_size=self.hparams.train_length,
                    hop_size=self.hparams.train_length // 2,
                    win_length=self.hparams.train_length,
                    n_mels=128,
                    w_sc=0.0,
                    device="cuda" if self.hparams.gpus > 0 else "cpu",
                )
            elif recon_loss == "melstft":
                self.recon_losses[recon_loss] = auraloss.freq.MelSTFTLoss(
                    self.hparams.sample_rate,
                    device="cuda" if self.hparams.gpus > 0 else "cpu",
                )
            elif recon_loss == "l1":
                self.recon_losses[recon_loss] = torch.nn.L1Loss()
            elif recon_loss == "sisdr":
                self.recon_losses[recon_loss] = auraloss.time.SISDRLoss()
            else:
                raise ValueError(
                    f"Invalid reconstruction loss: {self.hparams.recon_losses}"
                )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor = None,
        e_y: torch.Tensor = None,
        z: torch.Tensor = None,
        dsp_mode: DSPMode = DSPMode.NONE,
        analysis_length: int = 0,
        sample_rate: int = 24000,
    ):
        """Forward pass through the system subnetworks.

        Args:
            x (tensor): Input audio tensor with shape (batch x 1 x samples)
            y (tensor): Target audio tensor with shape (batch x 1 x samples)
            e_y (tensor): Target embedding with shape (batch x edim)
            z (tensor): Bottleneck latent.
            dsp_mode (DSPMode): Mode of operation for the DSP blocks.
            analysis_length (optional, int): Only analyze the first N samples.
            sample_rate (optional, int): Desired sampling rate for the DSP blocks.

        You must supply target audio `y`, `z`, or an embedding for the target `e_y`.

        Returns:
            y_hat (tensor): Output audio.
            p (tensor):
            e (tensor):

        """
        bs, chs, samp = x.size()

        if sample_rate != self.hparams.sample_rate:
            x_enc = torchaudio.transforms.Resample(
                sample_rate, self.hparams.sample_rate
            ).to(x.device)(x)
            if y is not None:
                y_enc = torchaudio.transforms.Resample(
                    sample_rate, self.hparams.sample_rate
                ).to(x.device)(y)
        else:
            x_enc = x
            y_enc = y

        if analysis_length > 0:
            x_enc = x_enc[..., :analysis_length]
            if y is not None:
                y_enc = y_enc[..., :analysis_length]

        e_x = self.encoder(x_enc)  # generate latent embedding for input

        if y is not None:
            e_y = self.encoder(y_enc)  # generate latent embedding for target
        elif e_y is None:
            raise RuntimeError("Must supply y, z, or e_y. None supplied.")

        # learnable comparision
        p = self.controller(e_x, e_y, z=z)

        # process audio conditioned on parameters
        # if there are multiple channels process them using same parameters
        y_hat = torch.zeros(x.shape).type_as(x)
        for ch_idx in range(chs):
            y_hat_ch = self.processor(
                x[:, ch_idx : ch_idx + 1, :],
                p,
                epsilon=self.eps_scheduler.epsilon,
                dsp_mode=dsp_mode,
                sample_rate=sample_rate,
            )
            y_hat[:, ch_idx : ch_idx + 1, :] = y_hat_ch

        return y_hat, p, e_x

    def common_paired_step(
        self,
        batch: Tuple,
        batch_idx: int,
        optimizer_idx: int = 0,
        train: bool = False,
    ):
        """Model step used for validation and training.

        Args:
            batch (Tuple[Tensor, Tensor]): Batch items containing input audio (x) and target audio (y).
            batch_idx (int): Index of the batch within the current epoch.
            optimizer_idx (int): Index of the optimizer, this step is called once for each optimizer.
                The firs optimizer corresponds to the generator and the second optimizer,
                corresponds to the adversarial loss (when in use).
            train (bool): Whether step is called during training (True) or validation (False).
        """
        x, y = batch
        loss = 0
        dsp_mode = self.hparams.dsp_mode

        if train and dsp_mode.INFER.name == DSPMode.INFER.name:
            dsp_mode = DSPMode.NONE

        # proces input audio through model
        if self.hparams.style_transfer:
            length = x.shape[-1]

            x_A = x[..., : length // 2]
            x_B = x[..., length // 2 :]

            y_A = y[..., : length // 2]
            y_B = y[..., length // 2 :]

            if torch.rand(1).sum() > 0.5:
                y_ref = y_B
                y = y_A
                x = x_A
            else:
                y_ref = y_A
                y = y_B
                x = x_B

            y_hat, p, e = self(x, y=y_ref, dsp_mode=dsp_mode)
        else:
            y_ref = None
            y_hat, p, e = self(x, dsp_mode=dsp_mode)

        # compute reconstruction loss terms
        for loss_idx, (loss_name, recon_loss_fn) in enumerate(
            self.recon_losses.items()
        ):
            temp_loss = recon_loss_fn(y_hat, y)  # reconstruction loss
            loss += float(self.hparams.recon_loss_weights[loss_idx]) * temp_loss

            self.log(
                ("train" if train else "val") + f"_loss/{loss_name}",
                temp_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )

        # log the overall aggregate loss
        self.log(
            ("train" if train else "val") + "_loss/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )

        # store audio data
        data_dict = {
            "x": x.cpu(),
            "y": y.cpu(),
            "p": p.cpu(),
            "e": e.cpu(),
            "y_hat": y_hat.cpu(),
        }

        if y_ref is not None:
            data_dict["y_ref"] = y_ref.cpu()

        return loss, data_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss, _ = self.common_paired_step(
            batch,
            batch_idx,
            optimizer_idx,
            train=True,
        )

        return loss

    def training_epoch_end(self, training_step_outputs):
        if self.hparams.spsa_schedule and self.hparams.processor_model == "spsa":
            self.eps_scheduler.step(
                self.trainer.callback_metrics[self.hparams.train_monitor],
            )

    def validation_step(self, batch, batch_idx):
        loss, data_dict = self.common_paired_step(batch, batch_idx)

        return data_dict

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if optimizer_idx == 0:
            optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        # we need additional optimizer for the discriminator
        optimizers = []
        g_optimizer = torch.optim.Adam(
            chain(
                self.encoder.parameters(),
                self.processor.parameters(),
                self.controller.parameters(),
            ),
            lr=self.hparams.lr,
            betas=(0.9, 0.999),
        )
        optimizers.append(g_optimizer)

        g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            g_optimizer,
            patience=self.hparams.lr_patience,
            verbose=True,
        )
        ms1 = int(self.hparams.max_epochs * 0.8)
        ms2 = int(self.hparams.max_epochs * 0.95)
        print(
            "Learning rate schedule:",
            f"0 {self.hparams.lr:0.2e} -> ",
            f"{ms1} {self.hparams.lr*0.1:0.2e} -> ",
            f"{ms2} {self.hparams.lr*0.01:0.2e}",
        )
        g_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            g_optimizer,
            milestones=[ms1, ms2],
            gamma=0.1,
        )

        lr_schedulers = {
            "scheduler": g_scheduler,
        }

        return optimizers, lr_schedulers

    def train_dataloader(self):

        train_dataset = AudioDataset(
            self.hparams.audio_dir,
            subset="train",
            train_frac=self.hparams.train_frac,
            half=self.hparams.half,
            length=self.hparams.train_length,
            input_dirs=self.hparams.input_dirs,
            random_scale_input=self.hparams.random_scale_input,
            random_scale_target=self.hparams.random_scale_target,
            buffer_size_gb=self.hparams.buffer_size_gb,
            buffer_reload_rate=self.hparams.buffer_reload_rate,
            num_examples_per_epoch=self.hparams.train_examples_per_epoch,
            augmentations={
                "pitch": {"sr": self.hparams.sample_rate},
                "tempo": {"sr": self.hparams.sample_rate},
            },
            freq_corrupt=self.hparams.freq_corrupt,
            drc_corrupt=self.hparams.drc_corrupt,
            ext=self.hparams.ext,
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
            persistent_workers=True,
            timeout=60,
        )

    def val_dataloader(self):

        val_dataset = AudioDataset(
            self.hparams.audio_dir,
            subset="val",
            half=self.hparams.half,
            train_frac=self.hparams.train_frac,
            length=self.hparams.val_length,
            input_dirs=self.hparams.input_dirs,
            buffer_size_gb=self.hparams.buffer_size_gb,
            buffer_reload_rate=self.hparams.buffer_reload_rate,
            random_scale_input=self.hparams.random_scale_input,
            random_scale_target=self.hparams.random_scale_target,
            num_examples_per_epoch=self.hparams.val_examples_per_epoch,
            augmentations={},
            freq_corrupt=self.hparams.freq_corrupt,
            drc_corrupt=self.hparams.drc_corrupt,
            ext=self.hparams.ext,
        )

        self.val_dataset = val_dataset

        g = torch.Generator()
        g.manual_seed(0)

        return torch.utils.data.DataLoader(
            val_dataset,
            num_workers=1,
            batch_size=self.hparams.batch_size,
            worker_init_fn=utils.seed_worker,
            generator=g,
            pin_memory=True,
            persistent_workers=True,
            timeout=60,
        )
    def shutdown(self):
        del self.processor

    # add any model hyperparameters here
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # --- Training  ---
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--lr_patience", type=int, default=20)
        parser.add_argument("--recon_losses", nargs="+", default=["l1"])
        parser.add_argument("--recon_loss_weights", nargs="+", default=[1.0])
        # --- Controller  ---
        parser.add_argument(
            "--processor_model",
            type=str,
            help="autodiff, spsa, tcn1, tcn2, proxy0, proxy1, proxy2",
        )
        parser.add_argument("--controller_hidden_dim", type=int, default=256)
        parser.add_argument("--style_transfer", action="store_true")
        # --- Encoder ---
        parser.add_argument("--encoder_model", type=str, default="mobilenet_v2")
        parser.add_argument("--encoder_embed_dim", type=int, default=128)
        parser.add_argument("--encoder_width_mult", type=int, default=2)
        parser.add_argument("--encoder_ckpt", type=str, default=None)
        parser.add_argument("--encoder_freeze", action="store_true", default=False)
        # --- TCN  ---
        parser.add_argument("--tcn_causal", action="store_true")
        parser.add_argument("--tcn_nblocks", type=int, default=4)
        parser.add_argument("--tcn_dilation_growth", type=int, default=8)
        parser.add_argument("--tcn_channel_width", type=int, default=32)
        parser.add_argument("--tcn_kernel_size", type=int, default=13)
        # ---  SPSA  ---
        parser.add_argument("--plugin_config_file", type=str, default=None)
        parser.add_argument("--spsa_epsilon", type=float, default=0.001)
        parser.add_argument("--spsa_schedule", action="store_true")
        parser.add_argument("--spsa_patience", type=int, default=10)
        parser.add_argument("--spsa_verbose", action="store_true")
        parser.add_argument("--spsa_factor", type=float, default=0.5)
        parser.add_argument("--spsa_parallel", action="store_true")
        # --- Proxy ----
        parser.add_argument("--proxy_ckpts", nargs="+")
        parser.add_argument("--freeze_proxies", action="store_true", default=False)
        parser.add_argument("--use_dsp", action="store_true", default=False)
        parser.add_argument("--dsp_mode", choices=DSPMode, type=DSPMode)
        # --- Dataset  ---
        parser.add_argument("--audio_dir", type=str)
        parser.add_argument("--ext", type=str, default="wav")
        parser.add_argument("--input_dirs", nargs="+")
        parser.add_argument("--buffer_reload_rate", type=int, default=1000)
        parser.add_argument("--buffer_size_gb", type=float, default=1.0)
        parser.add_argument("--sample_rate", type=int, default=24000)
        parser.add_argument("--dsp_sample_rate", type=int, default=24000)
        parser.add_argument("--shuffle", type=bool, default=True)
        parser.add_argument("--random_scale_input", action="store_true")
        parser.add_argument("--random_scale_target", action="store_true")
        parser.add_argument("--freq_corrupt", action="store_true")
        parser.add_argument("--drc_corrupt", action="store_true")
        parser.add_argument("--train_length", type=int, default=65536)
        parser.add_argument("--train_frac", type=float, default=0.8)
        parser.add_argument("--half", action="store_true")
        parser.add_argument("--train_examples_per_epoch", type=int, default=10000)
        parser.add_argument("--val_length", type=int, default=131072)
        parser.add_argument("--val_examples_per_epoch", type=int, default=1000)
        parser.add_argument("--num_workers", type=int, default=16)

        return parser
