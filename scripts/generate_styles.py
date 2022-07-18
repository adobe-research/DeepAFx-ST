import os
import sys
import glob
import torch
import random
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
from itertools import repeat
import pytorch_lightning as pl

from deepafx_st.processors.dsp.peq import parametric_eq
from deepafx_st.processors.dsp.compressor import compressor


def get_random_patch(x, sample_rate, length_samples):
    length = int(length_samples)
    silent = True
    while silent:
        start_idx = np.random.randint(0, x.shape[-1] - length - 1)
        stop_idx = start_idx + length
        x_crop = x[:, start_idx:stop_idx]

        # check for silence
        frames = length // sample_rate
        silent_frames = []
        for n in range(frames):
            start_idx = int(n * sample_rate)
            stop_idx = start_idx + sample_rate
            x_frame = x_crop[:, start_idx:stop_idx]
            if (x_frame ** 2).mean() > 3e-4:
                silent_frames.append(False)
            else:
                silent_frames.append(True)
        silent = True if any(silent_frames) else False

    x_crop /= x_crop.abs().max()

    return x_crop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_dir",
        help="Path to directory containing source audio.",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        help="Path to directory to save output audio.",
        type=str,
    )
    parser.add_argument(
        "--length_samples",
        help="Length of the output audio examples in samples.",
        type=float,
        default=131072,
    )
    parser.add_argument(
        "--lookahead_samples",
        help="Length of the processing lookahead.",
        type=float,
        default=16384,
    )
    parser.add_argument(
        "--num",
        help="Number of examples to generate from each style.",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--ext",
        help="Expected file extension for audio files.",
        type=str,
        default="wav",
    )

    args = parser.parse_args()
    pl.seed_everything(42)

    # find all audio files in directory
    dataset_name = os.path.basename(args.audio_dir)
    audio_filepaths = glob.glob(os.path.join(args.audio_dir, f"*.{args.ext}"))
    print(f"Found {len(audio_filepaths)} audio files.")
    if len(audio_filepaths) < 1:
        raise RuntimeError(f"No audio files found in {args.audio_dir}.")

    # split files into three subsets (train, val, test)
    random.shuffle(audio_filepaths)
    train_idx = int(len(audio_filepaths) * 0.6)
    val_idx = int(len(audio_filepaths) * 0.2)
    train_subset = audio_filepaths[:train_idx]
    val_subset = audio_filepaths[train_idx : train_idx + val_idx]
    test_subset = audio_filepaths[train_idx + val_idx :]
    print(
        f"Train ({len(train_subset)})  Val ({len(val_subset)})  Test ({len(test_subset)})"
    )

    subsets = {
        "train": train_subset,
        "val": val_subset,
        "test": test_subset,
    }

    # There are five different pre-defined styles
    styles = [
        "neutral",  # 1. Neutral - Presecnce + Normal compression
        "broadcast",  # 2. Broadcast - More Presence + Aggressive compression
        "telephone",  # 3. Telephone - Bandpass effect + compressor
        "bright",  # 4. Bright - Strong high-shelf filter
        "warm",  # 5. Warm - Bass boost with high-shelf decrease
    ]

    for style_idx, style in enumerate(styles):
        # reset the seed for each style
        # pl.seed_everything(42)
        print(f"Generating {style} ({style_idx+1}/{len(styles)}) style examples...")
        # generate examples
        subset_index_offset = 0

        for subset_name, subset_filepaths in subsets.items():
            # create output directory if needed
            style_dir = os.path.join(args.output_dir, subset_name, style)
            if not os.path.isdir(style_dir):
                os.makedirs(style_dir)

            # futher split the tracks for each style
            tracks_per_style = len(subset_filepaths) // len(styles)
            start_idx = style_idx * tracks_per_style
            stop_idx = start_idx + tracks_per_style
            print(start_idx, stop_idx)
            style_subset_filepaths = subset_filepaths[start_idx:stop_idx]

            if subset_name == "train":
                num_examples = int(args.num * 0.6)
            else:
                num_examples = int(args.num * 0.2)

            style_subset_filepaths = style_subset_filepaths * len(styles)

            # copy style subset filepaths to create desired number of examples
            if num_examples > len(style_subset_filepaths):
                style_subset_filepaths *= int(num_examples // len(style_subset_filepaths))
            else:
                style_subset_filepaths = style_subset_filepaths[:num_examples]

            for n, input_filepath in enumerate(tqdm(style_subset_filepaths, ncols=120)):
                x, sr = torchaudio.load(input_filepath)  # load file
                chs, samp = x.size()

                # get random audio patch
                x = get_random_patch(
                    x,
                    sr,
                    args.length_samples + args.lookahead_samples,
                )

                # add some randomized headroom
                headroom_db = (torch.rand(1) * 6) + 3
                x = x / x.abs().max()
                x *= 10 ** (-headroom_db / 20.0)

                # apply selected style
                if style == "neutral":
                    # ----------- compressor -------------
                    threshold = -((torch.rand(1) * 10.0).numpy().squeeze() + 20.0)
                    attack_sec = (torch.rand(1) * 0.020).numpy().squeeze() + 0.050
                    release_sec = (torch.rand(1) * 0.200).numpy().squeeze() + 0.100
                    ratio = (torch.rand(1) * 0.5).numpy().squeeze() + 1.5
                    # ----------- parametric eq -----------
                    low_shelf_gain_db = (torch.rand(1) * 2.0).numpy().squeeze() + 1.0
                    low_shelf_cutoff_freq = (torch.rand(1) * 120).numpy().squeeze() + 80
                    first_band_gain_db = 0.0
                    first_band_cutoff_freq = 1000.0
                    high_shelf_gain_db = (torch.rand(1) * 2.0).numpy().squeeze() + 1.0
                    high_shelf_cutoff_freq = (
                        torch.rand(1) * 2000
                    ).numpy().squeeze() + 6000
                elif style == "broadcast":
                    # ----------- compressor -------------
                    threshold = -((torch.rand(1) * 10).numpy().squeeze() + 40)
                    attack_sec = (torch.rand(1) * 0.025).numpy().squeeze() + 0.005
                    release_sec = (torch.rand(1) * 0.100).numpy().squeeze() + 0.050
                    ratio = (torch.rand(1) * 2.0).numpy().squeeze() + 3.0
                    # ----------- parametric eq -----------
                    low_shelf_gain_db = (torch.rand(1) * 4.0).numpy().squeeze() + 2.0
                    low_shelf_cutoff_freq = (torch.rand(1) * 120).numpy().squeeze() + 80
                    first_band_gain_db = 0.0
                    first_band_cutoff_freq = 1000.0
                    high_shelf_gain_db = (torch.rand(1) * 4.0).numpy().squeeze() + 2.0
                    high_shelf_cutoff_freq = (
                        torch.rand(1) * 2000
                    ).numpy().squeeze() + 6000
                elif style == "telephone":
                    # ----------- compressor -------------
                    threshold = -((torch.rand(1) * 20.0).numpy().squeeze() + 20)
                    attack_sec = (torch.rand(1) * 0.005).numpy().squeeze() + 0.001
                    release_sec = (torch.rand(1) * 0.050).numpy().squeeze() + 0.010
                    ratio = (torch.rand(1) * 1.5).numpy().squeeze() + 1.5
                    # ----------- parametric eq -----------
                    low_shelf_gain_db = -((torch.rand(1) * 6).numpy().squeeze() + 20)
                    low_shelf_cutoff_freq = (
                        torch.rand(1) * 200
                    ).numpy().squeeze() + 200
                    first_band_gain_db = (torch.rand(1) * 4).numpy().squeeze() + 12
                    first_band_cutoff_freq = (
                        torch.rand(1) * 1000
                    ).numpy().squeeze() + 1000
                    high_shelf_gain_db = -((torch.rand(1) * 6).numpy().squeeze() + 20)
                    high_shelf_cutoff_freq = (
                        torch.rand(1) * 2000
                    ).numpy().squeeze() + 4000
                elif style == "bright":
                    # ----------- compressor -------------
                    ratio = 1.0
                    threshold = 0.0
                    attack_sec = 0.050
                    release_sec = 0.250
                    # ----------- parametric eq -----------
                    low_shelf_gain_db = -((torch.rand(1) * 6).numpy().squeeze() + 20)
                    low_shelf_cutoff_freq = (
                        torch.rand(1) * 200
                    ).numpy().squeeze() + 200
                    first_band_gain_db = 0.0
                    first_band_cutoff_freq = 1000.0
                    high_shelf_gain_db = (torch.rand(1) * 6).numpy().squeeze() + 20
                    high_shelf_cutoff_freq = (
                        torch.rand(1) * 2000
                    ).numpy().squeeze() + 8000
                elif style == "warm":
                    # ----------- compressor -------------
                    ratio = 1.0
                    threshold = 0.0
                    attack_sec = 0.050
                    release_sec = 0.250
                    # ----------- parametric eq -----------
                    low_shelf_gain_db = (torch.rand(1) * 6).numpy().squeeze() + 20
                    low_shelf_cutoff_freq = (
                        torch.rand(1) * 200
                    ).numpy().squeeze() + 200
                    first_band_gain_db = 0.0
                    first_band_cutoff_freq = 1000.0
                    high_shelf_gain_db = -(torch.rand(1) * 6).numpy().squeeze() + 20
                    high_shelf_cutoff_freq = (
                        torch.rand(1) * 2000
                    ).numpy().squeeze() + 8000
                else:
                    raise RuntimeError(f"Invalid style: {style}.")

                # apply effects with parameters
                x_out = torch.zeros(x.shape).type_as(x)
                for ch_idx in range(chs):
                    x_peq_ch = parametric_eq(
                        x[ch_idx, :].view(-1).numpy(),
                        sr,
                        low_shelf_gain_dB=low_shelf_gain_db,
                        low_shelf_cutoff_freq=low_shelf_cutoff_freq,
                        first_band_gain_dB=first_band_gain_db,
                        first_band_cutoff_freq=first_band_cutoff_freq,
                        high_shelf_gain_dB=high_shelf_gain_db,
                        high_shelf_cutoff_freq=high_shelf_cutoff_freq,
                    )
                    x_comp_ch = compressor(
                        x_peq_ch,
                        sr,
                        threshold=threshold,
                        ratio=ratio,
                        attack_time=attack_sec,
                        release_time=release_sec,
                    )
                    x_out[ch_idx, :] = torch.tensor(x_comp_ch)

                # crop out lookahead
                x_out = x_out.view(chs, -1)
                x_out = x_out[:, args.lookahead_samples :]

                # peak normalize
                x_out /= x_out.abs().max()

                output_filename = f"{n+subset_index_offset:03d}_{style}_{dataset_name}_{subset_name}.wav"
                output_filepath = os.path.join(style_dir, output_filename)
                torchaudio.save(output_filepath, x_out, sr)
            subset_index_offset += n + 1
