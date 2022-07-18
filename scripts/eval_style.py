import os
import sys
import glob
import json
import torch
import auraloss
import argparse
import torchaudio
import numpy as np
import pytorch_lightning as pl
from multiprocessing import process

from deepafx_st.utils import DSPMode, seed_worker
from deepafx_st.system import System
from deepafx_st.data.dataset import AudioDataset
from deepafx_st.models.baselines import BaselineEQAndComp
from deepafx_st.metrics import (
    LoudnessError,
    RMSEnergyError,
    SpectralCentroidError,
    CrestFactorError,
    PESQ,
    MelSpectralDistance,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ckpt_dir",
        help="Top level directory containing model checkpoints to evaluate",
    )
    parser.add_argument(
        "--root_dir",
        default="/mnt/session_space",
        help="Top level directory containing datasets.",
    )
    parser.add_argument(
        "--real",
        help="Run real world evaluation. Otherwise synthetic.",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset to evaluate on. (vctk, daps, podcast, libritts, jamendo)",
        default="libritts",
    )
    parser.add_argument(
        "--dataset_dir",
        help="Path to root dataset directory",
        default="LibriTTS/train_clean_360_24000c",
    )
    parser.add_argument(
        "--output",
        help="Path to root directory to store outputs.",
        default="./",
    )
    parser.add_argument(
        "--length",
        help="Audio example length to use in samples.",
        type=int,
        default=131072,
    )
    parser.add_argument(
        "--gpu",
        help="Run models on GPU.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save",
        help="Save audio and plots for each example.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--examples",
        help="Number of examples to evaluate.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--subset",
        help="Evaluate on the train, val, or test sets.",
        default="val",
        type=str,
    )
    parser.add_argument(
        "--seed",
        help="Random seed.",
        default=42,
    )
    parser.add_argument(
        "--tcn1_version",
        help="Pre-trained TCN 1 model version.",
        default=0,
    )
    parser.add_argument(
        "--tcn2_version",
        help="Pre-trained TCN 2 model version.",
        default=0,
    )
    parser.add_argument(
        "--proxy0_version",
        help="Pre-trained Proxy 0 model version.",
        default=0,
    )
    parser.add_argument(
        "--proxy2_version",
        help="Pre-trained Proxy 2 model version.",
        default=0,
    )
    parser.add_argument(
        "--proxy0m_version",
        help="Pre-trained Proxy 0 model version.",
        default=0,
    )
    parser.add_argument(
        "--proxy2m_version",
        help="Pre-trained Proxy 2 model version.",
        default=0,
    )
    parser.add_argument(
        "--spsa_version",
        help="Pre-trained SPSA model version.",
        default=0,
    )
    parser.add_argument(
        "--autodiff_version",
        help="Pre-trained autodiff model version.",
        default=0,
    )
    parser.add_argument(
        "--checkpoint_loss",
        type=str,
        default="val",
        help="Evaluate on best 'train' or 'val' loss",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="wav",
        help="Dataset audio extension.",
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    pl.seed_everything(args.seed)

    eval_dir = os.path.join(args.output, f"eval_{args.dataset}")

    if not os.path.isdir(eval_dir):
        os.makedirs(eval_dir)

    if args.gpu:
        device = "cuda"
    else:
        device = "cpu"

    # ---- setup the dataset ----
    if not args.real:
        eval_dataset = AudioDataset(
            args.root_dir,
            subset=args.subset,
            half=False,
            train_frac=0.9,
            length=args.length,
            input_dirs=[args.dataset_dir],
            buffer_size_gb=2.0,
            buffer_reload_rate=args.examples,
            num_examples_per_epoch=args.examples,
            augmentations={},
            freq_corrupt=True,
            drc_corrupt=True,
            ext=args.ext,
        )
        sample_rate = eval_dataset.sample_rate
        # eval_dataset = torch.utils.data.DataLoader(
        #    eval_dataset,
        #    shuffle=False,
        #    num_workers=1,
        #    batch_size=1,
        #    worker_init_fn=seed_worker,
        #    pin_memory=True,
        #    persistent_workers=True,
        #    timeout=60,
        # )
    else:
        eval_dataset = None
    print(f"Dataset fs={sample_rate}")

    models = {}
    # --------------- setup pre-trained models ---------------
    for processor_model_id in [
        "tcn1",
        "tcn2",
        "spsa",
        "proxy0",
        "proxy1",
        "proxy2",
        "proxy0m",
        "proxy1m",
        "proxy2m",
        "autodiff",
    ]:

        if processor_model_id == "proxy1":
            processor_model_id_dir = "proxy0"
        elif processor_model_id == "proxy1m":
            processor_model_id_dir = "proxy0m"
        else:
            processor_model_id_dir = processor_model_id

        log_dir = os.path.join(
            args.ckpt_dir,
            processor_model_id_dir,
            "lightning_logs",
            f"""version_{getattr(args, f"{processor_model_id_dir}_version")}""",
        )
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        pckpt_dir = os.path.join(log_dir, "pretrained_checkpoints")
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        pckpts = glob.glob(os.path.join(pckpt_dir, "*.ckpt"))

        if len(ckpts) < 1:
            print(
                f"No {processor_model_id} checkpoint found in {ckpt_dir}. Skipping..."
            )
            continue
        else:
            ckpt = [c for c in ckpts if args.checkpoint_loss in c][0]

        print(f"Loading {processor_model_id} {ckpt} on {device}...")

        dsp_mode = DSPMode.NONE

        # search for pre-trained models
        if "m" in processor_model_id:
            peq_ckpt = "checkpoints/proxies/jamendo/peq/lightning_logs/version_0/checkpoints/epoch=326-step=204374-val-jamendo-peq.ckpt"
            comp_ckpt = "checkpoints/proxies/jamendo/comp/lightning_logs/version_0/checkpoints/epoch=274-step=171874-val-jamendo-comp.ckpt"
        else:
            peq_ckpt = "checkpoints/proxies/libritts/peq/lightning_logs/version_1/checkpoints/epoch=111-step=139999-val-libritts-peq.ckpt"
            comp_ckpt = "checkpoints/proxies/libritts/comp/lightning_logs/version_1/checkpoints/epoch=255-step=319999-val-libritts-comp.ckpt"

        if processor_model_id == "proxy0" or processor_model_id == "proxy0m":
            # peq_ckpt = [pc for pc in pckpts if "peq" in pc][0]
            # comp_ckpt = [pc for pc in pckpts if "comp" in pc][0]
            proxy_ckpts = [peq_ckpt, comp_ckpt]
            print(f"Found {len(proxy_ckpts)}: {proxy_ckpts}")
            model = (
                System.load_from_checkpoint(
                    ckpt,
                    dsp_mode=DSPMode.NONE,
                    proxy_ckpts=proxy_ckpts,
                    dsp_sample_rate=sample_rate,
                    strict=False,
                )
                .eval()
                .to(device)
            )
            dsp_mode = DSPMode.NONE
            processor_model_name = processor_model_id
        elif processor_model_id == "proxy1" or processor_model_id == "proxy1m":
            # peq_ckpt = [pc for pc in pckpts if "peq" in pc][0]
            # comp_ckpt = [pc for pc in pckpts if "comp" in pc][0]
            proxy_ckpts = [peq_ckpt, comp_ckpt]
            print(f"Found {len(proxy_ckpts)}: {proxy_ckpts}")
            model = (
                System.load_from_checkpoint(
                    ckpt,
                    use_dsp=DSPMode.INFER,
                    proxy_ckpts=proxy_ckpts,
                    dsp_sample_rate=sample_rate,
                    strict=False,
                )
                .eval()
                .to(device)
            )
            processor_model_name = processor_model_id
            model.hparams.dsp_mode = DSPMode.INFER
        elif processor_model_id == "proxy2" or processor_model_id == "proxy2m":
            # peq_ckpt = [pc for pc in pckpts if "peq" in pc][0]
            # comp_ckpt = [pc for pc in pckpts if "comp" in pc][0]
            proxy_ckpts = [peq_ckpt, comp_ckpt]
            print(f"Found {len(proxy_ckpts)}: {proxy_ckpts}")
            model = (
                System.load_from_checkpoint(
                    ckpt,
                    dsp_mode=DSPMode.INFER,
                    proxy_ckpts=proxy_ckpts,
                    dsp_sample_rate=sample_rate,
                    strict=False,
                )
                .eval()
                .to(device)
            )
            processor_model_name = processor_model_id
            model.hparams.dsp_mode = DSPMode.INFER
        elif processor_model_id == "tcn1":
            processor_model_name = "TCN1"
            model = (
                System.load_from_checkpoint(
                    ckpt,
                    dsp_mode=DSPMode.INFER,
                    dsp_sample_rate=sample_rate,
                    strict=False,
                )
                .eval()
                .to(device)
            )
        elif processor_model_id == "tcn2":
            processor_model_name = "TCN2"
            model = (
                System.load_from_checkpoint(
                    ckpt,
                    dsp_mode=DSPMode.INFER,
                    dsp_sample_rate=sample_rate,
                    strict=False,
                )
                .eval()
                .to(device)
            )
        elif processor_model_id == "autodiff":
            processor_model_name = "Autodiff"
            model = (
                System.load_from_checkpoint(
                    ckpt,
                    dsp_mode=DSPMode.NONE,
                    dsp_sample_rate=sample_rate,
                    strict=False,
                )
                .eval()
                .to(device)
            )
        elif processor_model_id == "spsa":
            processor_model_name = "SPSA"
            model = (
                System.load_from_checkpoint(
                    ckpt,
                    dsp_mode=DSPMode.NONE,
                    dsp_sample_rate=sample_rate,
                    strict=False,
                    spsa_parallel=False,
                )
                .eval()
                .to(device)
            )
        else:
            raise RuntimeError(f"Unexpected processor_model_id: {processor_model_id}")

        models[f"{processor_model_name} ({args.dataset})"] = model

    if len(list(models.keys())) < 1:
        raise ValueError("No checkpoints found for evaluation. Exiting...")

    # create the baseline model
    baseline_model = BaselineEQAndComp(sample_rate=sample_rate)
    models[f"Baseline ({args.dataset})"] = baseline_model

    # ---- setup the metrics ----
    metrics = {
        "PESQ": PESQ(sample_rate),
        "MRSTFT": auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[32, 128, 512, 2048, 8192, 32768],
            hop_sizes=[16, 64, 256, 1024, 4096, 16384],
            win_lengths=[32, 128, 512, 2048, 8192, 32768],
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        ),
        "MSD": MelSpectralDistance(sample_rate),
        "SCE": SpectralCentroidError(sample_rate),
        "CFE": CrestFactorError(),
        "RMS": RMSEnergyError(),
        "LUFS": LoudnessError(sample_rate),
    }
    metrics_dict = {"Corrupt": {}}

    # ---- start the evaluation ----
    for bidx, batch in enumerate(eval_dataset, 0):
        x, y = batch

        x = x.to(device)
        y = y.to(device)

        # sum to mono
        x = x.mean(0, keepdim=True)
        y = y.mean(0, keepdim=True)
        # print(x.shape, y.shape)

        # split inputs in half for style transfer
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

        # corrupted input peak normalized to -3 dBFS
        # x_norm = x / x.abs().max()
        # x_norm *= 10 ** (-12.0 / 20)

        # compute metrics with the corrupt input
        for metric_name, metric in metrics.items():
            if metric_name not in metrics_dict["Corrupt"]:
                metrics_dict["Corrupt"][metric_name] = []

            try:
                val = metric(x.cpu().view(1, 1, -1), y.cpu().view(1, 1, -1))
            except:
                val = -1
            metrics_dict["Corrupt"][metric_name].append(val)

        outputs = {}
        # now iterate over models and compute metrics
        for model_name, model in models.items():
            if model_name not in metrics_dict:
                metrics_dict[model_name] = {}

            # forward pass through model
            with torch.no_grad():
                if "Baseline" in model_name:
                    y_hat = model(
                        x.cpu().view(1, 1, -1).clone(),
                        y_ref.cpu().view(1, 1, -1).clone(),
                    )
                else:
                    y_hat, p, e = model(
                        x.view(1, 1, -1).clone(),
                        y=y_ref.view(1, 1, -1).clone(),
                        dsp_mode=model.hparams.dsp_mode,
                        sample_rate=sample_rate,
                        analysis_length=131072,
                    )

            y_hat = y_hat.cpu()
            y = y.cpu()
            outputs[model_name] = y_hat  # store

            # compute all metrics
            for metric_name, metric in metrics.items():
                if metric_name not in metrics_dict[model_name]:
                    metrics_dict[model_name][metric_name] = []

                try:
                    val = metric(y_hat.view(1, 1, -1), y.view(1, 1, -1))
                except:
                    val = -1
                metrics_dict[model_name][metric_name].append(val)

            if args.save:
                y_hat_filepath = os.path.join(
                    eval_dir, f"{bidx:04d}_{model_name}_y_hat.wav"
                )
                torchaudio.save(y_hat_filepath, y_hat.view(1, -1), sample_rate)

        if args.save:
            x_filepath = os.path.join(eval_dir, f"{bidx:04d}_x.wav")
            y_filepath = os.path.join(eval_dir, f"{bidx:04d}_y.wav")
            torchaudio.save(x_filepath, x.view(1, -1).cpu(), sample_rate)
            torchaudio.save(y_filepath, y.view(1, -1).cpu(), sample_rate)

        print(bidx + 1)
        for model_name, model_metrics in metrics_dict.items():
            sys.stdout.write(f"\n {model_name:22} ")
            for metric_name, metric_list in model_metrics.items():
                sys.stdout.write(f"{metric_name}: {np.mean(metric_list):0.3f}  ")
            sys.stdout.flush()
        print()
        print("-" * 32)

        if bidx + 1 == args.examples:
            print("Evaluation complete.")
            json_metrics_dict = {}
            for model_name, model_metrics in metrics_dict.items():
                if model_name not in json_metrics_dict:
                    json_metrics_dict[model_name] = {}
                for metric_name, metric_list in model_metrics.items():
                    if metric_name not in json_metrics_dict:
                        sanitized_metric_list = []
                        for elm in metric_list:
                            if isinstance(elm, torch.Tensor):
                                sanitized_metric_list.append(elm.numpy().tolist())
                            else:
                                sanitized_metric_list.append(elm)

                        json_metrics_dict[model_name][
                            metric_name
                        ] = sanitized_metric_list
            with open(
                os.path.join(
                    eval_dir,
                    f"{args.dataset.lower()}_{args.checkpoint_loss}_results.json",
                ),
                "w",
            ) as fp:
                json.dump(json_metrics_dict, fp, indent=True)

            for model_name, model in models.items():
                del model

            break
