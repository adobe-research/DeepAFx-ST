import os
import sys
import glob
import torch
import auraloss
import argparse
import torchaudio
import numpy as np
import scipy.signal
import matplotlib
import pyloudnorm as pyln
import matplotlib.pyplot as plt

from deepafx_st.utils import DSPMode
from deepafx_st.utils import get_random_patch
from deepafx_st.processors.dsp.peq import biqaud
from deepafx_st.system import System


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
    w, h = scipy.signal.sosfreqz(sos, fs=22050, worN=2048)

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
    param_text=True,
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

    if param_text:
        ax.text(
            0,
            text_height,
            f"{threshold:0.1f} dB  {ratio:0.1f}:1",
            fontsize="small",
        )


def loudness_normalize(x, target_loudness=-24.0):
    x = x.view(1, -1)
    stereo_audio = x.repeat(2, 1).permute(1, 0).numpy()
    loudness = meter.integrated_loudness(stereo_audio)
    norm_x = pyln.normalize.loudness(
        stereo_audio,
        loudness,
        target_loudness,
    )
    x = torch.tensor(norm_x).permute(1, 0)
    x = x[0, :].view(1, -1)

    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to pre-trained system checkpoint.",
    )
    parser.add_argument(
        "--input_filepaths",
        help="List of audio filepaths for style transfer.",
        type=str,
        nargs="+",
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
        "--output_dir",
        type=str,
        help="Path to save audio outputs.",
        default="style_transfer_bulk",
    )
    parser.add_argument(
        "--target_loudness",
        type=float,
        help="Target audio output loudness in dB LUFS",
        default=-23.0,
    )
    parser.add_argument(
        "--num_interp_steps",
        type=int,
        help="Number of steps between each interpolated style.",
        default=4,
    )

    args = parser.parse_args()
    torch.manual_seed(42)

    device = "cuda" if args.gpu else "cpu"

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

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
    meter = pyln.Meter(sample_rate)  # Loudness meter

    # ----------- Plotting setup -------------

    colors = {
        "neutral": (70 / 255, 181 / 255, 211 / 255),  # neutral
        "broadcast": (52 / 255, 57 / 255, 60 / 255),  # broadcast
        "telephone": (219 / 255, 73 / 255, 76 / 255),  # telephone
        "warm": (235 / 255, 164 / 255, 50 / 255),  # warm
        "bright": (134 / 255, 170 / 255, 109 / 255),  # bright
    }

    # ----------- Locate audio files  -------------
    for input_filepath in args.input_filepaths:
        outputs = {}  # store the transformed style outputs
        interp_outputs = []  # style interpolations

        # create one plot for each input style
        fig, axs = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)

        input_style_name = os.path.basename(os.path.dirname(input_filepath))
        input_style_color = colors[input_style_name]
        x, x_sr = torchaudio.load(input_filepath)
        x = x / x.abs().max()

        output_filepath = os.path.join(
            args.output_dir,
            f"style_transfer_{input_style_name}.wav",
        )
        torchaudio.save(
            output_filepath,
            loudness_normalize(x, args.target_loudness),
            x_sr,
        )

        x *= 10 ** (-12.0 / 20.0)

        # use all other styles are targets
        style_filepaths = list(args.input_filepaths)
        style_filepaths.remove(input_filepath)

        # ----------- interpolate between all target styles -----------
        target_style_names = []
        for sidx, style_filepath in enumerate(style_filepaths):
            target_style_a_name = os.path.basename(os.path.dirname(style_filepath))
            target_style_names.append(target_style_a_name)
            target_style_a_color = colors[target_style_a_name]
            y_a, y_a_sr = torchaudio.load(style_filepath)
            y_a = y_a / y_a.abs().max()
            y_a *= 10 ** (-12.0 / 20.0)

            # get the next style in list
            next_sidx = sidx + 1
            if next_sidx > len(style_filepaths) - 1:
                next_sidx = 0
            style_filepath = style_filepaths[next_sidx]
            target_style_b_name = os.path.basename(os.path.dirname(style_filepath))
            target_style_b_color = colors[target_style_b_name]
            y_b, y_b_sr = torchaudio.load(style_filepath)
            y_b = y_b / y_b.abs().max()
            y_b *= 10 ** (-12.0 / 20.0)

            # compute style embeddings
            with torch.no_grad():
                style_a_embed = system.encoder(y_a.view(1, 1, -1))
                style_b_embed = system.encoder(y_b.view(1, 1, -1))

            # repeat the input audio for more length
            x_long = x.repeat(1, 4)

            # linear interpolation between style embeddings
            for w_idx, w in enumerate(np.linspace(0, 1, args.num_interp_steps)):
                style_embed = (w * style_b_embed) + ((1 - w) * style_a_embed)
                print(w_idx, style_embed)

                # run our model
                with torch.no_grad():
                    y_hat_system, p, e_system = system(
                        x_long.view(1, 1, -1),
                        e_y=style_embed,
                        analysis_length=131072,
                    )

                interp_outputs.append(
                    loudness_normalize(
                        y_hat_system.view(1, -1),
                        args.target_loudness,
                    ),
                )

        # chop outputs into an interpolation
        num_frames = args.num_interp_steps * len(style_filepaths)
        frame_size = x_long.shape[-1] // num_frames
        tmp_output = torch.zeros(x_long.shape)

        fade_size = 4096

        for n in range(num_frames):
            start_idx = (n * frame_size) - fade_size
            stop_idx = (start_idx + frame_size) + fade_size
            if start_idx < 0:
                start_idx = 0
            if stop_idx > tmp_output.shape[-1]:
                stop_idx = tmp_output.shape[-1] - 1
            frame_audio = interp_outputs[n][:, start_idx:stop_idx]
            # apply linear fade in and out
            ramp_up = np.linspace(0, 1, num=fade_size)
            ramp_down = np.linspace(1, 0, num=fade_size)
            frame_audio[:, :fade_size] *= ramp_up
            frame_audio[:, -fade_size:] *= ramp_down
            tmp_output[:, start_idx:stop_idx] = frame_audio

        filename = (
            f"style_transfer_{input_style_name}_to_"
            + "_".join(target_style_names)
            + ".wav"
        )
        output_filepath = os.path.join(args.output_dir, filename)

        # normalize to target loudness
        torchaudio.save(
            output_filepath,
            loudness_normalize(tmp_output, args.target_loudness),
            x_sr,
        )

        # ----------- single transfer to each style (with plotting) -----------
        for sidx, style_filepath in enumerate(style_filepaths):
            target_style_name = os.path.basename(os.path.dirname(style_filepath))
            target_style_color = colors[target_style_name]
            y, y_sr = torchaudio.load(style_filepath)
            y = y / y.abs().max()
            y *= 10 ** (-12.0 / 20.0)

            # run our model
            with torch.no_grad():
                y_hat_system, p, e_system = system(
                    x.view(1, 1, -1),
                    y=y.view(1, 1, -1),
                )

            # normalize and store
            y_hat_system /= y_hat_system.abs().max()
            transfer_name = f"{input_style_name}_to_{target_style_name}"
            outputs[transfer_name] = y_hat_system.view(1, -1)

            # -------- split params between EQ and Comp. --------
            p_peq = p[:, : system.processor.peq.num_control_params]
            p_comp = p[:, system.processor.peq.num_control_params :]

            p_peq_denorm = system.processor.peq.denormalize_params(p_peq.view(-1))
            p_peq_denorm = [p.numpy() for p in p_peq_denorm]

            p_comp_denorm = system.processor.comp.denormalize_params(p_comp.view(-1))
            p_comp_denorm = [p.numpy() for p in p_comp_denorm]

            comp_params = {}

            # -------- Create Frequency response plot --------
            plot_peq_response(
                p_peq_denorm,
                sample_rate,
                ax=axs[0],
                label=target_style_name,
                color=colors[target_style_name],
                center_line=True if sidx == 0 else False,
            )

            plot_comp_response(
                p_comp_denorm,
                sample_rate,
                ax=axs[1],
                label=target_style_name,
                color=target_style_color,
                center_line=True if sidx == 0 else False,
                param_text=True,
            )

        if args.save:
            for output_name, output_audio in outputs.items():
                output_filepath = os.path.join(
                    args.output_dir,
                    f"style_transfer_{output_name}.wav",
                )

                # normalize to target loudness
                torchaudio.save(
                    output_filepath,
                    loudness_normalize(output_audio, args.target_loudness),
                    y_sr,
                )

        # --------- formating for Parametric EQ ---------=
        plot_filepath = os.path.join(
            args.output_dir,
            f"style_transfer_{input_style_name}",
        )
        plt.title(f"{input_style_name} as input")
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
        axs[0].tick_params(axis="x", which="minor", colors="lightgray", labelcolor="k")
        axs[0].tick_params(axis="x", which="major", colors="lightgray", labelcolor="k")
        axs[0].tick_params(axis="y", which="major", colors="lightgray", labelcolor="k")
        axs[0].legend(
            ncol=4,
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
        axs[1].tick_params(axis="x", which="major", colors="lightgray", labelcolor="k")
        axs[1].tick_params(axis="y", which="major", colors="lightgray", labelcolor="k")
        axs[1].set(adjustable="box", aspect="equal")

        plt.tight_layout()
        plt.savefig(plot_filepath + ".png", dpi=300)
        plt.savefig(plot_filepath + ".svg")
        plt.savefig(plot_filepath + ".pdf")
