import os
import sys
import glob
from types import resolve_bases
import torch
import auraloss
import argparse
import torchaudio
import numpy as np
import scipy.signal
import matplotlib
import matplotlib.pyplot as plt

from deepafx_st.utils import DSPMode
from deepafx_st.utils import loudness_normalize
from deepafx_st.processors.dsp.peq import biqaud, parametric_eq
from deepafx_st.processors.dsp.compressor import compressor
from deepafx_st.system import System
from deepafx_st.models.baselines import BaselineEQAndComp
from deepafx_st.metrics import (
    LoudnessError,
    RMSEnergyError,
    SpectralCentroidError,
    CrestFactorError,
    PESQ,
    MelSpectralDistance,
)


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

    if label == "telephone":
        label = "Tele"

    # measure freq response
    w, h = scipy.signal.sosfreqz(sos, fs=22050, worN=2048)

    if ax is None:
        fig, axs = plt.subplots()

    if center_line:
        ax.plot(w, np.zeros(w.shape), color="lightgray")

    (handle,) = ax.plot(
        w, 20 * np.log10(np.abs(h)), label=label.capitalize(), color=color
    )
    if points:
        ax.scatter(ls_freq, ls_gain, color=color)
        ax.scatter(f1_freq, f1_gain, color=color)
        ax.scatter(f2_freq, f2_gain, color=color)
        ax.scatter(f3_freq, f3_gain, color=color)
        ax.scatter(f4_freq, f4_gain, color=color)
        ax.scatter(hs_freq, hs_gain, color=color)

    return handle


def plot_comp_response(
    p_comp_denorm,
    sr,
    ax=None,
    label=None,
    color=None,
    center_line=False,
    prev_height=None,
    plot_idx=0,
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

    text_height = threshold + ((-20 - threshold) / (ratio)) - 1

    # text_height += style_idx * 2
    # if ((style_idx + 1) % 2) == 0:
    text_width = -18
    # else:
    #    text_width = -14

    # plot the first part of the line
    ax.plot(x, y, color=color)
    if center_line:
        ax.plot(x, x, color="lightgray", linestyle="--")
        # ax.text(-19, -22, f"Thres.  Ratio")
        ax.text(-19, -22, f"Ratio")

    # ax.text(-18, text_height, f"{threshold:0.1f}   {ratio:0.1f}")
    ax.text(text_width, text_height, f"{ratio:0.1f}")

    return text_height


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to pre-trained system checkpoint.",
    )
    parser.add_argument(
        "--input_filepaths",
        type=str,
        help="List of input audio filepaths.",
        nargs="+",
    )
    parser.add_argument(
        "--style_filepaths",
        help="List of style audio filepaths.",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--gpu",
        help="Run System on GPU.",
        action="store_true",
    )
    parser.add_argument(
        "--modify_input",
        help="Apply increasing strong effects to input.",
        action="store_true",
    )
    parser.add_argument(
        "--save",
        help="Save audio examples.",
        action="store_true",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to save audio outputs.",
        default="style_transfer",
    )
    parser.add_argument(
        "--target_loudness",
        type=float,
        help="Target audio output loudness in dB LUFS",
        default=-23.0,
    )

    args = parser.parse_args()
    torch.manual_seed(42)

    device = "cuda" if args.gpu else "cpu"

    fontlist = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
    fontlist = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    # print(fontlist)
    # set font
    plt.rcParams["font.family"] = "Nimbus Roman"

    # --------------- setup pre-trained model ---------------
    use_dsp = DSPMode.NONE
    system = System.load_from_checkpoint(
        args.ckpt_path,
        use_dsp=use_dsp,
        batch_size=1,
        spsa_parallel=False,
    )
    system.eval()
    if args.gpu:
        system.to("cuda")

    sample_rate = system.hparams.sample_rate

    # create the baseline model
    baseline_model = BaselineEQAndComp(sample_rate=sample_rate)

    colors = {
        "neutral": (70 / 255, 181 / 255, 211 / 255),  # neutral
        "broadcast": (52 / 255, 57 / 255, 60 / 255),  # broadcast
        "telephone": (219 / 255, 73 / 255, 76 / 255),  # telephone
        "warm": (235 / 255, 164 / 255, 50 / 255),  # warm
        "bright": (134 / 255, 170 / 255, 109 / 255),  # bright
    }

    for input_filepath in args.input_filepaths:
        # normalize input
        input_name = os.path.basename(input_filepath).replace(".wav", "")
        x, x_sr = torchaudio.load(input_filepath)

        if x_sr != sample_rate:
            x = torchaudio.transforms.Resample(x_sr, sample_rate)(x)

        x = x[:, : 262144 * 2]
        x /= x.abs().max()
        print(x.shape)

        fig, axs = plt.subplots(figsize=(5, 2), nrows=1, ncols=2)

        # fig, axs = plt.subplots(figsize=(4, 5), nrows=2, ncols=1)
        cmap = matplotlib.cm.get_cmap("viridis")
        handles = []
        prev_height = None

        if args.modify_input:
            parameters = {
                "high_shelf_gain_dB": [0.0, 1.0, 3.0, 6.0, 12.0],  # , 18.0],
                "threshold_dB": [-3.0, -12.0, -24.0, -40.0, -62.0],  # , -70],
                "ratio": [1.0, 2.0, 3.0, 3.0, 4.0],  # , 8.0],
            }
            args.style_filepaths *= len(parameters["high_shelf_gain_dB"])

        for style_idx, style_filepath in enumerate(args.style_filepaths):
            y, y_sr = torchaudio.load(style_filepath)
            style_name = os.path.basename(os.path.dirname(style_filepath))

            # apply increasing effects
            if args.modify_input:
                hsg = parameters["high_shelf_gain_dB"][style_idx]
                thr = parameters["threshold_dB"][style_idx]
                rto = parameters["ratio"][style_idx]
                x_proc = parametric_eq(
                    x.view(-1).numpy(),
                    float(sample_rate),
                    high_shelf_gain_dB=hsg,
                    high_shelf_cutoff_freq=4000.0,
                )
                x_proc = compressor(
                    x_proc,
                    float(sample_rate),
                    threshold=thr,
                    ratio=rto,
                    attack_time=0.005,
                    release_time=0.050,
                    knee_dB=0.0,
                )
                x_proc = torch.tensor(x_proc).view(1, -1)
            else:
                x_proc = x.clone()

            x_norm = x_proc / x_proc.abs().max()
            x_norm *= 10 ** (-6.0 / 20.0)

            x_proc = x_proc[:, -131072:]

            # normalize reference
            y_norm = y / y.abs().max()
            y_norm *= 10 ** (-12.0 / 20.0)

            # run our model
            with torch.no_grad():
                y_hat_system, p, e_system = system(
                    x_norm.view(1, 1, -1),
                    y=y_norm.view(1, 1, -1),
                    analysis_length=131072,
                )

            # -------- split params between EQ and Comp. --------
            p_peq = p[:, : system.processor.peq.num_control_params]
            p_comp = p[:, system.processor.peq.num_control_params :]

            p_peq_denorm = system.processor.peq.denormalize_params(p_peq.view(-1))
            p_peq_denorm = [p.numpy() for p in p_peq_denorm]

            p_comp_denorm = system.processor.comp.denormalize_params(p_comp.view(-1))
            p_comp_denorm = [p.numpy() for p in p_comp_denorm]

            # comp_params = {}

            # -------- Create Frequency response plot --------
            if args.modify_input:
                label = f"{style_idx}"
                color = cmap(style_idx / len(parameters["high_shelf_gain_dB"]))

            else:
                label = style_name
                color = colors[style_name]

            handle = plot_peq_response(
                p_peq_denorm,
                sample_rate,
                ax=axs[0],
                label=label,
                color=color,
                center_line=True if style_idx == 0 else False,
            )
            handles.append(handle)

            prev_height = plot_comp_response(
                p_comp_denorm,
                sample_rate,
                ax=axs[1],
                label=label,
                color=color,
                center_line=True if style_idx == 0 else False,
                prev_height=prev_height,
                plot_idx=style_idx,
            )

            if args.save:
                if not os.path.isdir(args.output_dir):
                    os.makedirs(args.output_dir)

                input_filepath = os.path.join(
                    args.output_dir,
                    f"{style_idx}_{input_name}.wav",
                )
                style_filepath = os.path.join(
                    args.output_dir,
                    f"{style_idx}_{style_name}.wav",
                )
                system_filepath = os.path.join(
                    args.output_dir,
                    f"{style_idx}_{input_name}_to_{style_name}_system.wav",
                )

                torchaudio.save(
                    input_filepath,
                    loudness_normalize(
                        x_norm,
                        sample_rate,
                        args.target_loudness,
                    ),
                    sample_rate,
                )
                torchaudio.save(
                    style_filepath,
                    loudness_normalize(
                        y_norm,
                        sample_rate,
                        args.target_loudness,
                    ),
                    sample_rate,
                )
                torchaudio.save(
                    system_filepath,
                    loudness_normalize(
                        y_hat_system.view(1, -1),
                        sample_rate,
                        args.target_loudness,
                    ),
                    sample_rate,
                )

            plot_filepath = os.path.join(
                args.output_dir, f"style_transfer_{input_name}"
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
            axis="x",
            which="minor",
            colors="lightgray",
            labelcolor="k",
        )
        axs[0].tick_params(
            axis="x",
            which="major",
            colors="lightgray",
            labelcolor="k",
        )
        axs[0].tick_params(
            axis="y",
            which="major",
            colors="lightgray",
            labelcolor="k",
        )
        if args.modify_input:
            ncol = 5
        else:
            ncol = 5

        axs[0].legend(
            handles=handles,
            ncol=ncol,
            loc="upper center",
            columnspacing=0.8,
            framealpha=0.0,
            bbox_to_anchor=(1.05, 1.3),
            # bbox_to_anchor=(0.5, -0.025),
        )
        # axs[0].set(adjustable="box", aspect="auto")
        # --------- formating for compressor curve ---------
        axs[1].set_ylim([-80, -20])
        axs[1].set_xlim([-80, -20])
        axs[1].grid(c="lightgray", which="major")
        axs[1].spines["right"].set_visible(False)
        axs[1].spines["left"].set_visible(False)
        axs[1].spines["top"].set_visible(False)
        axs[1].spines["bottom"].set_visible(False)
        axs[1].set_ylabel("Output (dB)")
        axs[1].set_xlabel("Input (dB)")
        axs[1].tick_params(axis="x", which="major", colors="lightgray", labelcolor="k")
        axs[1].tick_params(axis="y", which="major", colors="lightgray", labelcolor="k")
        axs[1].set(adjustable="box", aspect="equal")

        # fig.tight_layout()
        fig.subplots_adjust(top=0.86, bottom=0.22, wspace=0.25, hspace=0.4, right=0.90)
        plt.savefig(plot_filepath + ".png", dpi=300)
        plt.savefig(plot_filepath + ".svg")
        plt.savefig(plot_filepath + ".pdf")
