# Adapted from:
# https://github.com/csteinmetz1/micro-tcn/blob/main/microtcn/utils.py
import os
import csv
import torch
import fnmatch
import numpy as np
import random
from enum import Enum
import pyloudnorm as pyln


class DSPMode(Enum):
    NONE = "none"
    TRAIN_INFER = "train_infer"
    INFER = "infer"

    def __str__(self):
        return self.value


def loudness_normalize(x, sample_rate, target_loudness=-24.0):
    x = x.view(1, -1)
    stereo_audio = x.repeat(2, 1).permute(1, 0).numpy()
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(stereo_audio)
    norm_x = pyln.normalize.loudness(
        stereo_audio,
        loudness,
        target_loudness,
    )
    x = torch.tensor(norm_x).permute(1, 0)
    x = x[0, :].view(1, -1)

    return x


def get_random_file_id(keys):
    # generate a random index into the keys of the input files
    rand_input_idx = torch.randint(0, len(keys) - 1, [1])[0]
    # find the key (file_id) correponding to the random index
    rand_input_file_id = list(keys)[rand_input_idx]

    return rand_input_file_id


def get_random_patch(audio_file, length, check_silence=True):
    silent = True
    while silent:
        start_idx = int(torch.rand(1) * (audio_file.num_frames - length))
        stop_idx = start_idx + length
        patch = audio_file.audio[:, start_idx:stop_idx].clone().detach()
        if (patch ** 2).mean() > 1e-4 or not check_silence:
            silent = False

    return start_idx, stop_idx


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def getFilesPath(directory, extension):

    n_path = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, extension):
                n_path.append(os.path.join(path, name))
    n_path.sort()

    return n_path


def count_parameters(model, trainable_only=True):

    if trainable_only:
        if len(list(model.parameters())) > 0:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            params = 0
    else:
        if len(list(model.parameters())) > 0:
            params = sum(p.numel() for p in model.parameters())
        else:
            params = 0

    return params


def system_summary(system):
    print(f"Encoder: {count_parameters(system.encoder)/1e6:0.2f} M")
    print(f"Processor: {count_parameters(system.processor)/1e6:0.2f} M")

    if hasattr(system, "adv_loss_fn"):
        for idx, disc in enumerate(system.adv_loss_fn.discriminators):
            print(f"Discriminator {idx+1}: {count_parameters(disc)/1e6:0.2f} M")


def center_crop(x, length: int):
    if x.shape[-1] != length:
        start = (x.shape[-1] - length) // 2
        stop = start + length
        x = x[..., start:stop]
    return x


def causal_crop(x, length: int):
    if x.shape[-1] != length:
        stop = x.shape[-1] - 1
        start = stop - length
        x = x[..., start:stop]
    return x


def denormalize(norm_val, max_val, min_val):
    return (norm_val * (max_val - min_val)) + min_val


def normalize(denorm_val, max_val, min_val):
    return (denorm_val - min_val) / (max_val - min_val)


def get_random_patch(audio_file, length, energy_treshold=1e-4):
    """Produce sample indicies for a random patch of size `length`.

    This function will check the energy of the selected patch to
    ensure that it is not complete silence. If silence is found,
    it will continue searching for a non-silent patch.

    Args:
        audio_file (AudioFile): Audio file object.
        length (int): Number of samples in random patch.

    Returns:
        start_idx (int): Starting sample index
        stop_idx (int): Stop sample index
    """

    silent = True
    while silent:
        start_idx = int(torch.rand(1) * (audio_file.num_frames - length))
        stop_idx = start_idx + length
        patch = audio_file.audio[:, start_idx:stop_idx]
        if (patch ** 2).mean() > energy_treshold:
            silent = False

    return start_idx, stop_idx


def split_dataset(file_list, subset, train_frac):
    """Given a list of files, split into train/val/test sets.

    Args:
        file_list (list): List of audio files.
        subset (str): One of "train", "val", or "test".
        train_frac (float): Fraction of the dataset to use for training.

    Returns:
        file_list (list): List of audio files corresponding to subset.
    """
    assert train_frac > 0.1 and train_frac < 1.0

    total_num_examples = len(file_list)

    train_num_examples = int(total_num_examples * train_frac)
    val_num_examples = int(total_num_examples * (1 - train_frac) / 2)
    test_num_examples = total_num_examples - (train_num_examples + val_num_examples)

    if train_num_examples < 0:
        raise ValueError(
            f"No examples in training set. Try increasing train_frac: {train_frac}."
        )
    elif val_num_examples < 0:
        raise ValueError(
            f"No examples in validation set. Try decreasing train_frac: {train_frac}."
        )
    elif test_num_examples < 0:
        raise ValueError(
            f"No examples in test set. Try decreasing train_frac: {train_frac}."
        )

    if subset == "train":
        start_idx = 0
        stop_idx = train_num_examples
    elif subset == "val":
        start_idx = train_num_examples
        stop_idx = start_idx + val_num_examples
    elif subset == "test":
        start_idx = train_num_examples + val_num_examples
        stop_idx = start_idx + test_num_examples + 1
    else:
        raise ValueError("Invalid subset: {subset}.")

    return file_list[start_idx:stop_idx]


def rademacher(size):
    """Generates random samples from a Rademacher distribution +-1

    Args:
        size (int):

    """
    m = torch.distributions.binomial.Binomial(1, 0.5)
    x = m.sample(size)
    x[x == 0] = -1
    return x


def get_subset(csv_file):
    subset_files = []
    with open(csv_file) as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            subset_files.append(row["filepath"])

    return list(set(subset_files))


def conform_length(x: torch.Tensor, length: int):
    """Crop or pad input on last dim to match `length`."""
    if x.shape[-1] < length:
        padsize = length - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, padsize))
    elif x.shape[-1] > length:
        x = x[..., :length]

    return x


def linear_fade(
    x: torch.Tensor,
    fade_ms: float = 50.0,
    sample_rate: float = 22050,
):
    """Apply fade in and fade out to last dim."""
    fade_samples = int(fade_ms * 1e-3 * 22050)

    fade_in = torch.linspace(0.0, 1.0, steps=fade_samples)
    fade_out = torch.linspace(1.0, 0.0, steps=fade_samples)

    # fade in
    x[..., :fade_samples] *= fade_in

    # fade out
    x[..., -fade_samples:] *= fade_out

    return x


# def get_random_patch(x, sample_rate, length_samples):
#     length = length_samples
#     silent = True
#     while silent:
#         start_idx = np.random.randint(0, x.shape[-1] - length - 1)
#         stop_idx = start_idx + length
#         x_crop = x[0:1, start_idx:stop_idx]

#         # check for silence
#         frames = length // sample_rate
#         silent_frames = []
#         for n in range(frames):
#             start_idx = n * sample_rate
#             stop_idx = start_idx + sample_rate
#             x_frame = x_crop[0:1, start_idx:stop_idx]
#             if (x_frame ** 2).mean() > 3e-4:
#                 silent_frames.append(False)
#             else:
#                 silent_frames.append(True)
#         silent = True if any(silent_frames) else False

#     x_crop /= x_crop.abs().max()

#     return x_crop
