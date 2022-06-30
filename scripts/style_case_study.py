import os
import sys
import glob
import torch
import auraloss
import itertools
import argparse
import torchaudio
import numpy as np
import scipy.signal
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from tqdm import tqdm

from deepafx_st.utils import DSPMode
from deepafx_st.utils import get_random_patch
from deepafx_st.system import System
from deepafx_st.models.baselines import BaselineEQAndComp
from deepafx_st.processors.dsp.peq import biqaud
from deepafx_st.metrics import (
    LoudnessError,
    RMSEnergyError,
    SpectralCentroidError,
    CrestFactorError,
    PESQ,
    MelSpectralDistance,
)

colors = {
    "neutral": (70 / 255, 181 / 255, 211 / 255),  # neutral
    "broadcast": (52 / 255, 57 / 255, 60 / 255),  # broadcast
    "telephone": (219 / 255, 73 / 255, 76 / 255),  # telephone
    "warm": (235 / 255, 164 / 255, 50 / 255),  # warm
    "bright": (134 / 255, 170 / 255, 109 / 255),  # bright
}


def plot_peq_response(
    p_peq_denorm,
    sr,
    ax=None,
    label=None,
    color=None,
    points=False,
    center_line=False,
):

    ls_gain = p_peq_denorm[0]
    ls_freq = p_peq_denorm[1]
    ls_q = p_peq_denorm[2]
    b0, a0 = biqaud(ls_gain, ls_freq, ls_q, sr, filter_type="low_shelf")
    sos0 = np.concatenate((b0, a0))

    f1_gain = p_peq_denorm[3]
    f1_freq = p_peq_denorm[4]
    f1_q = p_peq_denorm[5]
    b1, a1 = biqaud(f1_gain, f1_freq, f1_q, sr, filter_type="peaking")
    sos1 = np.concatenate((b1, a1))

    f2_gain = p_peq_denorm[6]
    f2_freq = p_peq_denorm[7]
    f2_q = p_peq_denorm[8]
    b2, a2 = biqaud(f2_gain, f2_freq, f2_q, sr, filter_type="peaking")
    sos2 = np.concatenate((b2, a2))

    f3_gain = p_peq_denorm[9]
    f3_freq = p_peq_denorm[10]
    f3_q = p_peq_denorm[11]
    b3, a3 = biqaud(f3_gain, f3_freq, f3_q, sr, filter_type="peaking")
    sos3 = np.concatenate((b3, a3))

    f4_gain = p_peq_denorm[12]
    f4_freq = p_peq_denorm[13]
    f4_q = p_peq_denorm[14]
    b4, a4 = biqaud(f4_gain, f4_freq, f4_q, sr, filter_type="peaking")
    sos4 = np.concatenate((b4, a4))

    hs_gain = p_peq_denorm[15]
    hs_freq = p_peq_denorm[16]
    hs_q = p_peq_denorm[17]
    b5, a5 = biqaud(hs_gain, hs_freq, hs_q, sr, filter_type="high_shelf")
    sos5 = np.concatenate((b5, a5))

    sos = [sos0, sos1, sos2, sos3, sos4, sos5]
    sos = np.array(sos)
    # print(sos.shape)
    # print(sos)

    # measure freq response
    w, h = scipy.signal.sosfreqz(sos, fs=sr, worN=2048)

    if ax is None:
        fig, axs = plt.subplots()

    if center_line:
        ax.plot(w, np.zeros(w.shape), color="lightgray")

    ax.plot(w, 20 * np.log10(np.abs(h)), label=label, color=color)
    if points:
        ax.scatter(ls_freq, ls_gain, color=color)
        ax.scatter(f1_freq, f1_gain, color=color)
        ax.scatter(f2_freq, f2_gain, color=color)
        ax.scatter(f3_freq, f3_gain, color=color)
        ax.scatter(f4_freq, f4_gain, color=color)
        ax.scatter(hs_freq, hs_gain, color=color)


def plot_comp_response(
    p_comp_denorm,
    sr,
    ax=None,
    label=None,
    color=None,
    center_line=False,
):

    # get parameters
    threshold = p_comp_denorm[0]
    ratio = p_comp_denorm[1]
    attack_ms = p_comp_denorm[2] * 1000
    release_ms = p_comp_denorm[3] * 1000
    knee_db = p_comp_denorm[4]
    makeup_db = p_comp_denorm[5]

    # print(knee_db)

    x = np.linspace(-80, 0)  # input level
    y = np.zeros(x.shape)  # output level

    idx = np.where((2 * (x - threshold)) < -knee_db)
    y[idx] = x[idx]

    idx = np.where((2 * np.abs(x - threshold)) <= knee_db)
    y[idx] = x[idx] + (
        (1 / ratio - 1) * (((x[idx] - threshold + (knee_db / 2))) ** 2)
    ) / (2 * knee_db)

    idx = np.where((2 * (x - threshold)) > knee_db)
    y[idx] = threshold + ((x[idx] - threshold) / (ratio))

    text_height = threshold + ((0 - threshold) / (ratio))

    # plot the first part of the line
    ax.plot(x, y, label=label, color=color)
    if center_line:
        ax.plot(x, x, color="lightgray", linestyle="--")
    ax.text(0, text_height, f"{threshold:0.1f} dB  {ratio:0.1f}:1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_paths",
        type=str,
        help="Path to pre-trained system checkpoints.",
        nargs="+",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of style transfer to perform for each style transfer example.",
    )
    parser.add_argument(
        "--style_audio",
        help="List of style audio filepaths.",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        help="Run System on GPU.",
        action="store_true",
    )
    parser.add_argument(
        "--save",
        help="Save audio examples.",
        action="store_true",
    )
    parser.add_argument(
        "--plot",
        help="Save parameter prediction plots.",
        action="store_true",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to save audio outputs.",
        default="style_case_study",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        help="Input audio sample rate.",
        default=24000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed.",
        default=16,
    )

    args = parser.parse_args()
    pl.seed_everything(args.seed)
    device = "cuda" if args.gpu else "cpu"
    metrics_dict = {"Corrupt": {}, "Baseline": {}}

    # --------------- setup pre-trained model ---------------
    models = {}
    peq_ckpt = "/import/c4dm-datasets/deepafx_st/logs/proxies/libritts/peq/lightning_logs/version_1/checkpoints/epoch=111-step=139999-val-libritts-peq.ckpt"
    comp_ckpt = "/import/c4dm-datasets/deepafx_st/logs/proxies/libritts/comp/lightning_logs/version_1/checkpoints/epoch=255-step=319999-val-libritts-comp.ckpt"

    for ckpt_path in args.ckpt_paths:
        model_name = os.path.basename(ckpt_path).replace(".ckpt", "")

        if "proxy" in model_name:
            use_dsp = DSPMode.INFER
        else:
            use_dsp = DSPMode.NONE

        system = System.load_from_checkpoint(
            ckpt_path,
            use_dsp=use_dsp,
            batch_size=1,
            spsa_parallel=False,
            proxy_ckpts=[peq_ckpt, comp_ckpt],
            strict=False,
        )
        system.eval()
        if args.gpu:
            system.to("cuda")
        models[model_name] = system

        metrics_dict[model_name] = {}

    # create the baseline model
    baseline_model = BaselineEQAndComp(sample_rate=args.sample_rate)

    # ---- setup the metrics ----
    metrics = {
        # "PESQ": PESQ(sample_rate),
        # "MRSTFT": auraloss.freq.MultiResolutionSTFTLoss(
        #   fft_sizes=[32, 128, 512, 2048, 8192, 32768],
        #    hop_sizes=[16, 64, 256, 1024, 4096, 16384],
        #    win_lengths=[32, 128, 512, 2048, 8192, 32768],
        #    w_sc=0.0,
        #    w_phs=0.0,
        #    w_lin_mag=1.0,
        #    w_log_mag=1.0,
        # ),
        "MSD": MelSpectralDistance(args.sample_rate),
        "SCE": SpectralCentroidError(args.sample_rate),
        # "CFE": CrestFactorError(),
        "RMS": RMSEnergyError(),
        "LUFS": LoudnessError(args.sample_rate),
    }

    # ----------- load and pre-process audio -------------
    style_dirs = glob.glob(os.path.join(args.style_audio, "*"))
    style_dirs = [sd for sd in style_dirs if os.path.isdir(sd)]

    transfers = itertools.product(style_dirs, style_dirs)

    metrics_overall = {}
    for transfer in transfers:
        input_style_dir, target_style_dir = transfer
        input_style = os.path.basename(input_style_dir)
        target_style = os.path.basename(target_style_dir)
        style_transfer_name = f"{input_style}-->{target_style}"
        transfer_output_dir = os.path.join(args.output_dir, style_transfer_name)
        if not os.path.isdir(transfer_output_dir):
            os.makedirs(transfer_output_dir)
        print(style_transfer_name)
        metrics_dict[style_transfer_name] = {}

        # get all examples from the input style
        input_filepaths = glob.glob(os.path.join(input_style_dir, "*.wav"))

        # get all examples from the target style
        target_filepaths = glob.glob(os.path.join(target_style_dir, "*.wav"))

        for n in tqdm(range(args.num_examples), ncols=80):
            input_filepath = np.random.choice(input_filepaths)
            target_filepath = np.random.choice(target_filepaths)
            input_name = os.path.basename(input_filepath).replace("*.wav", "")
            target_name = os.path.basename(target_filepath).replace("*.wav", "")
            x, x_sr = torchaudio.load(input_filepath)
            y, y_sr = torchaudio.load(target_filepath)

            chs, samp = x.size()

            # normalize
            x_norm = x / x.abs().max()
            x_norm *= 10 ** (-12.0 / 20.0)
            y_norm = y / y.abs().max()
            y_norm *= 10 ** (-12.0 / 20.0)

            if args.gpu:
                x_norm = x_norm.to("cuda")
                y_norm = y_norm.to("cuda")

            # ------------------ compute model metrics ------------------
            # run our models
            model_outputs = {}
            model_params = {}
            for model_name, system in models.items():
                with torch.no_grad():
                    y_hat_system, p, e_system = system(
                        x_norm.view(1, 1, -1),
                        y=y_norm.view(1, 1, -1),
                        dsp_mode=system.hparams.use_dsp,
                        analysis_length=131072,
                        sample_rate=x_sr,
                    )

                short_model_name = model_name.split("-")[-1]

                # normalize
                # y_hat_system = y_hat_system / y_hat_system.abs().max()
                # y_hat_system *= 10 ** (-12.0 / 20.0)

                # ----------- store predicted audio and parameters -----------
                autodiff_key = [key for key in models.keys() if "autodiff" in key][0]
                tmp_system = models[autodiff_key]
                model_outputs[short_model_name] = y_hat_system  # store audio

                p_peq = p[:, : tmp_system.processor.peq.num_control_params].cpu()
                p_comp = p[:, tmp_system.processor.peq.num_control_params :].cpu()

                p_peq_denorm = tmp_system.processor.peq.denormalize_params(
                    p_peq.view(-1)
                )
                p_peq_denorm = [p.numpy() for p in p_peq_denorm]

                p_comp_denorm = tmp_system.processor.comp.denormalize_params(
                    p_comp.view(-1)
                )
                p_comp_denorm = [p.numpy() for p in p_comp_denorm]

                model_params[short_model_name] = {}  # store parameters
                model_params[short_model_name]["p_peq_denorm"] = p_peq_denorm
                model_params[short_model_name]["p_comp_denorm"] = p_comp_denorm

                # ----------- compute metrics -----------
                if short_model_name not in metrics_dict[style_transfer_name]:
                    metrics_dict[style_transfer_name][short_model_name] = {}

                for metric_name, metric_fn in metrics.items():
                    system_val = metric_fn(
                        y_hat_system[..., 16384:].cpu(),
                        y_norm.view(1, 1, -1)[..., 16384:].cpu(),
                    )

                    if (
                        metric_name
                        not in metrics_dict[style_transfer_name][short_model_name]
                    ):
                        metrics_dict[style_transfer_name][short_model_name][
                            metric_name
                        ] = []
                    metrics_dict[style_transfer_name][short_model_name][
                        metric_name
                    ].append(system_val)

            # ----------------- compute baseline metrics ------------------
            # run the baseline model
            y_hat_baseline = baseline_model(
                x_norm.view(1, 1, -1).cpu(),
                y_norm.view(1, 1, -1).cpu(),
            )

            if "Baseline" not in metrics_dict[style_transfer_name]:
                metrics_dict[style_transfer_name]["Baseline"] = {}

            for metric_name, metric_fn in metrics.items():
                baseline_val = metric_fn(
                    y_hat_baseline.cpu(), y_norm.view(1, 1, -1).cpu()
                )

                if metric_name not in metrics_dict[style_transfer_name]["Baseline"]:
                    metrics_dict[style_transfer_name]["Baseline"][metric_name] = []
                metrics_dict[style_transfer_name]["Baseline"][metric_name].append(
                    baseline_val
                )

            if "Corrupt" not in metrics_dict[style_transfer_name]:
                metrics_dict[style_transfer_name]["Corrupt"] = {}

            for metric_name, metric_fn in metrics.items():
                baseline_val = metric_fn(
                    x_norm.view(1, 1, -1).cpu(), y_norm.view(1, 1, -1).cpu()
                )

                if metric_name not in metrics_dict[style_transfer_name]["Corrupt"]:
                    metrics_dict[style_transfer_name]["Corrupt"][metric_name] = []
                metrics_dict[style_transfer_name]["Corrupt"][metric_name].append(
                    baseline_val
                )

            if args.save:
                input_filepath = os.path.join(
                    transfer_output_dir,
                    f"{n}_{input_style}-->{target_style}_input.wav",
                )
                target_filepath = os.path.join(
                    transfer_output_dir,
                    f"{n}_{input_style}-->{target_style}_target.wav",
                )
                baseline_filepath = os.path.join(
                    transfer_output_dir,
                    f"{n}_{input_style}-->{target_style}_baseline.wav",
                )

                for short_model_name, model_output in model_outputs.items():
                    model_filepath = os.path.join(
                        transfer_output_dir,
                        f"{n}_{input_style}-->{target_style}_{short_model_name}.wav",
                    )
                    torchaudio.save(
                        model_filepath,
                        model_output.view(chs, -1).cpu(),
                        y_sr,
                    )

                torchaudio.save(input_filepath, x_norm.cpu(), y_sr)
                torchaudio.save(target_filepath, y_norm.cpu(), y_sr)
                torchaudio.save(
                    baseline_filepath,
                    y_hat_baseline.view(chs, -1).cpu(),
                    y_sr,
                )

            if args.plot:
                # create main figure
                fig, axs = plt.subplots(figsize=(8, 3), nrows=1, ncols=2)

                for model_idx, (short_model_name, p) in enumerate(model_params.items()):

                    p_peq_denorm = p["p_peq_denorm"]
                    p_comp_denorm = p["p_comp_denorm"]

                    # -------- Create Frequency response plot --------
                    plot_peq_response(
                        p_peq_denorm,
                        args.sample_rate,
                        ax=axs[0],
                        label=short_model_name,
                        color=list(colors.values())[model_idx],
                        center_line=True if model_idx == 0 else False,
                    )

                    # -------- Create Compressor response plot --------
                    plot_comp_response(
                        p_comp_denorm,
                        args.sample_rate,
                        ax=axs[1],
                        label=short_model_name,
                        color=list(colors.values())[model_idx],
                        center_line=True if model_idx == 0 else False,
                    )

                plot_filepath = os.path.join(
                    transfer_output_dir,
                    f"{n}_{input_style}-->{target_style}",
                )

                # --------- formating for Parametric EQ ---------
                axs[0].set_ylim([-24, 24])
                axs[0].set_xlim([10, 10000])
                axs[0].set_xscale("log")
                axs[0].grid(c="lightgray", which="major")
                axs[0].grid(c="lightgray", which="minor")
                axs[0].set_ylabel("Magnitude (dB)")
                axs[0].set_xlabel("Frequency (Hz)")
                axs[0].spines["right"].set_visible(False)
                axs[0].spines["left"].set_visible(False)
                axs[0].spines["top"].set_visible(False)
                axs[0].spines["bottom"].set_visible(False)
                axs[0].tick_params(
                    axis="x", which="minor", colors="lightgray", labelcolor="k"
                )
                axs[0].tick_params(
                    axis="x", which="major", colors="lightgray", labelcolor="k"
                )
                axs[0].tick_params(
                    axis="y", which="major", colors="lightgray", labelcolor="k"
                )
                axs[0].legend(
                    ncol=6,
                    loc="lower center",
                    columnspacing=0.8,
                    framealpha=1.0,
                    bbox_to_anchor=(0.5, 1.05),
                )
                # --------- formating for compressor curve ---------
                axs[1].set_ylim([-80, 0])
                axs[1].set_xlim([-80, 0])
                axs[1].grid(c="lightgray", which="major")
                axs[1].spines["right"].set_visible(False)
                axs[1].spines["left"].set_visible(False)
                axs[1].spines["top"].set_visible(False)
                axs[1].spines["bottom"].set_visible(False)
                axs[1].set_ylabel("Output (dB)")
                axs[1].set_xlabel("Input (dB)")
                axs[1].set_title(f"{input_style} --> {target_style}")
                axs[1].tick_params(
                    axis="x", which="major", colors="lightgray", labelcolor="k"
                )
                axs[1].tick_params(
                    axis="y", which="major", colors="lightgray", labelcolor="k"
                )
                axs[1].set(adjustable="box", aspect="equal")

                plt.tight_layout()
                plt.savefig(plot_filepath + ".png", dpi=300)
                plt.savefig(plot_filepath + ".svg")
                plt.savefig(plot_filepath + ".pdf")
                plt.close("all")

        for model_name, model_metrics in metrics_dict[style_transfer_name].items():
            if model_name not in metrics_overall:
                metrics_overall[model_name] = {}
            sys.stdout.write(f"{model_name.ljust(10)} ")
            for metric_name, metric_values in model_metrics.items():
                mean_val = np.mean(metric_values)

                if metric_name not in metrics_overall[model_name]:
                    metrics_overall[model_name][metric_name] = []
                metrics_overall[model_name][metric_name].append(mean_val)

                sys.stdout.write(f"{metric_name}: {mean_val:0.3f}  ")
            print()
        print()

    print("----- Averages ----")
    for model_name, model_metrics in metrics_overall.items():
        sys.stdout.write(f"{model_name.ljust(10)} ")
        for metric_name, metric_values in model_metrics.items():
            mean_val = np.mean(metric_values)
            sys.stdout.write(f"{metric_name}: {mean_val:0.3f}  ")

    sys.exit(0)
