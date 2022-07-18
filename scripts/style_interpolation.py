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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to pre-trained system checkpoint.",
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        help="Path to input audio file.",
    )
    parser.add_argument(
        "--input_length",
        type=int,
        help="Number of seconds to process from input file.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10,
        help="Number of interpolation steps.",
    )
    parser.add_argument(
        "--style_a",
        help="Starting style.",
        type=str,
    )
    parser.add_argument(
        "--style_a_name",
        help="Starting style name.",
        type=str,
        default="a",
    )
    parser.add_argument(
        "--style_b",
        help="Ending style.",
        type=str,
    )
    parser.add_argument(
        "--style_b_name",
        help="Ending style name.",
        type=str,
        default="b",
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
        default="style_interpolation",
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

    # ----------- load and pre-process audio -------------
    # normalize input
    input_name = os.path.basename(args.input_audio).replace(".wav", "")
    x, x_sr = torchaudio.load(args.input_audio)
    input_length_samp = int(x_sr * args.input_length)
    # x = x[:, input_length_samp : input_length_samp * 2]
    # if x.shape[-1] > 131072:
    #    x = get_random_patch(x, x_sr, 131072)

    x_norm = x / x.abs().max()
    x_norm *= 10 ** (-12.0 / 20.0)

    fig, axs = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)
    cmap = matplotlib.cm.get_cmap("viridis")

    colors = {
        "podcast": (70 / 255, 181 / 255, 211 / 255),  # podcast
        "radio": (52 / 255, 57 / 255, 60 / 255),  # radio
        "telephone": (219 / 255, 73 / 255, 76 / 255),  # telephone
        "warm": (235 / 255, 164 / 255, 50 / 255),  # warm
        "bright": (134 / 255, 170 / 255, 109 / 255),  # bright
    }

    style_a_color = colors[args.style_a_name]
    style_b_color = colors[args.style_b_name]

    # compute start and ending style embeddings
    style_a_audio, sr = torchaudio.load(args.style_a)
    style_b_audio, sr = torchaudio.load(args.style_b)

    style_a_audio = style_a_audio / style_a_audio.abs().max()
    style_a_audio *= 10 ** (-12.0 / 20.0)
    style_b_audio = style_b_audio / style_b_audio.abs().max()
    style_b_audio *= 10 ** (-12.0 / 20.0)

    with torch.no_grad():
        style_a_embed = system.encoder(style_a_audio.view(1, 1, -1))
        style_b_embed = system.encoder(style_b_audio.view(1, 1, -1))

    # linear interpolation between style embeddings
    outputs = []
    for w_idx, w in enumerate(np.linspace(0, 1, args.num_steps)):

        style_embed = (w * style_b_embed) + ((1 - w) * style_a_embed)
        print(w_idx, style_embed)

        # run our model
        with torch.no_grad():
            y_hat_system, p, e_system = system(
                x_norm.view(1, 1, -1),
                e_y=style_embed,
                analysis_length=131072,
            )
        outputs.append(y_hat_system.view(-1))

        # -------- split params between EQ and Comp. --------
        p_peq = p[:, : system.processor.peq.num_control_params]
        p_comp = p[:, system.processor.peq.num_control_params :]

        p_peq_denorm = system.processor.peq.denormalize_params(p_peq.view(-1))
        p_peq_denorm = [p.numpy() for p in p_peq_denorm]

        p_comp_denorm = system.processor.comp.denormalize_params(p_comp.view(-1))
        p_comp_denorm = [p.numpy() for p in p_comp_denorm]

        comp_params = {}

        # -------- Create Frequency response plot --------
        if w_idx == 0:
            label = args.style_a_name
        elif w_idx == (args.num_steps - 1):
            label = args.style_b_name
        else:
            label = None

        # linear interpolkate RGB color
        style_color_R = (w * style_b_color[0]) + ((1 - w) * style_a_color[0])
        style_color_G = (w * style_b_color[1]) + ((1 - w) * style_a_color[1])
        style_color_B = (w * style_b_color[2]) + ((1 - w) * style_a_color[2])
        style_color = (style_color_R, style_color_G, style_color_B)

        plot_peq_response(
            p_peq_denorm,
            sample_rate,
            ax=axs[0],
            label=label,
            color=style_color,
            center_line=True if w_idx == 0 else False,
        )

        plot_comp_response(
            p_comp_denorm,
            sample_rate,
            ax=axs[1],
            label=label,
            color=style_color,
            center_line=True if w_idx == 0 else False,
            param_text=True if w_idx == 0 or (w_idx + 1) == args.num_steps else False,
        )

        if False:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)

            input_filepath = os.path.join(args.output_dir, f"{n}_{input_name}.wav")
            style_filepath = os.path.join(args.output_dir, f"{n}_{style_name}.wav")
            system_filepath = os.path.join(
                args.output_dir,
                f"{n}_{input_name}_{style_name}_system.wav",
            )
            baseline_filepath = os.path.join(
                args.output_dir,
                f"{n}_{input_name}_{style_name}_baseline.wav",
            )

            torchaudio.save(input_filepath, x_norm, y_sr)
            torchaudio.save(style_filepath, y_norm, y_sr)
            torchaudio.save(system_filepath, y_hat_system.view(1, -1), y_sr)
            # torchaudio.save(baseline_filepath, y_hat_baseline.view(1, -1), y_sr)

    # --------- Morphing audio style ---------
    frame_size = x.shape[-1] // args.num_steps
    output = torch.zeros(x.shape[-1])

    for n in range(args.num_steps):
        start_idx = n * frame_size
        stop_idx = start_idx + frame_size
        output[start_idx:stop_idx] = outputs[n][start_idx:stop_idx]

    audio_filepath = os.path.join(
        args.output_dir,
        f"style_interpolation_{args.style_a_name}_to_{args.style_b_name}_interp.wav",
    )
    torchaudio.save(audio_filepath, output.view(1, -1), sample_rate)

    # --------- Save input/output/style audio  ---------
    input_filepath = os.path.join(
        args.output_dir,
        f"style_interpolation_{args.style_a_name}_to_{args.style_b_name}_input.wav",
    )
    style_a_filepath = os.path.join(
        args.output_dir,
        f"style_interpolation_{args.style_a_name}_to_{args.style_b_name}_style={args.style_a_name}.wav",
    )
    style_b_filepath = os.path.join(
        args.output_dir,
        f"style_interpolation_{args.style_a_name}_to_{args.style_b_name}_style={args.style_b_name}.wav",
    )
    torchaudio.save(input_filepath, x.view(1, -1), sample_rate)
    torchaudio.save(style_a_filepath, style_a_audio.view(1, -1), sample_rate)
    torchaudio.save(style_b_filepath, style_b_audio.view(1, -1), sample_rate)

    # --------- formating for Parametric EQ ---------=
    plot_filepath = os.path.join(
        args.output_dir,
        f"style_interpolation_{args.style_a_name}_to_{args.style_b_name}",
    )
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
        ncol=2,
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
