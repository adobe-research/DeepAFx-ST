import torch
import torchaudio
import numpy as np


def gain(xs, min_dB=-12, max_dB=12):

    gain_dB = (torch.rand(1) * (max_dB - min_dB)) + min_dB
    gain_ln = 10 ** (gain_dB / 20)

    for idx, x in enumerate(xs):
        xs[idx] = x * gain_ln

    return xs


def peaking_filter(xs, sr=44100, frequency=1000, width_q=0.707, gain_db=12):

    # gain_db = ((torch.rand(1) * 6) + 6).numpy().squeeze()
    # width_q = (torch.rand(1) * 4).numpy().squeeze()
    # frequency = ((torch.rand(1) * 9960) + 40).numpy().squeeze()

    # if torch.rand(1) > 0.5:
    #    gain_db = -gain_db

    effects = [["equalizer", f"{frequency}", f"{width_q}", f"{gain_db}"]]

    for idx, x in enumerate(xs):
        y, sr = torchaudio.sox_effects.apply_effects_tensor(
            x, sr, effects, channels_first=True
        )
        xs[idx] = y

    return xs


def pitch_shift(xs, min_shift=-200, max_shift=200, sr=44100):

    shift = min_shift + (torch.rand(1)).numpy().squeeze() * (max_shift - min_shift)

    effects = [["pitch", f"{shift}"]]

    for idx, x in enumerate(xs):
        y, sr = torchaudio.sox_effects.apply_effects_tensor(
            x, sr, effects, channels_first=True
        )
        xs[idx] = y

    return xs


def time_stretch(xs, min_stretch=0.8, max_stretch=1.2, sr=44100):

    stretch = min_stretch + (torch.rand(1)).numpy().squeeze() * (
        max_stretch - min_stretch
    )

    effects = [["tempo", f"{stretch}"]]
    for idx, x in enumerate(xs):
        y, sr = torchaudio.sox_effects.apply_effects_tensor(
            x, sr, effects, channels_first=True
        )
        xs[idx] = y

    return xs


def frequency_corruption(xs, sr=44100):

    effects = []

    # apply a random number of peaking bands from 0 to 4s
    bands = [[200, 2000], [800, 4000], [2000, 8000], [4000, int((sr // 2) * 0.9)]]
    total_gain_db = 0.0
    for band in bands:
        if torch.rand(1).sum() > 0.2:
            frequency = (torch.randint(band[0], band[1], [1])).numpy().squeeze()
            width_q = ((torch.rand(1) * 10) + 0.1).numpy().squeeze()
            gain_db = ((torch.rand(1) * 48)).numpy().squeeze()

            if torch.rand(1).sum() > 0.5:
                gain_db = -gain_db

            total_gain_db += gain_db

            if np.abs(total_gain_db) >= 24:
                continue

            cmd = ["equalizer", f"{frequency}", f"{width_q}", f"{gain_db}"]
            effects.append(cmd)

    # low shelf (bass)
    if torch.rand(1).sum() > 0.2:
        gain_db = ((torch.rand(1) * 24)).numpy().squeeze()
        frequency = (torch.randint(20, 200, [1])).numpy().squeeze()
        if torch.rand(1).sum() > 0.5:
            gain_db = -gain_db
        effects.append(["bass", f"{gain_db}", f"{frequency}"])

    # high shelf (treble)
    if torch.rand(1).sum() > 0.2:
        gain_db = ((torch.rand(1) * 24)).numpy().squeeze()
        frequency = (torch.randint(4000, int((sr // 2) * 0.9), [1])).numpy().squeeze()
        if torch.rand(1).sum() > 0.5:
            gain_db = -gain_db
        effects.append(["treble", f"{gain_db}", f"{frequency}"])

    for idx, x in enumerate(xs):
        y, sr = torchaudio.sox_effects.apply_effects_tensor(
            x.view(1, -1) * 10 ** (-48 / 20), sr, effects, channels_first=True
        )
        # apply gain back
        y *= 10 ** (48 / 20)

        xs[idx] = y

    return xs


def dynamic_range_corruption(xs, sr=44100):
    """Apply an expander."""

    attack = (torch.rand([1]).numpy()[0] * 0.05) + 0.001
    release = (torch.rand([1]).numpy()[0] * 0.2) + attack
    knee = (torch.rand([1]).numpy()[0] * 12) + 0.0

    # design the compressor transfer function
    start = -100.0
    threshold = -(
        (torch.rand([1]).numpy()[0] * 20) + 10
    )  # threshold from -30 to -10 dB
    ratio = (torch.rand([1]).numpy()[0] * 4.0) + 1  # ratio from 1:1 to 5:1

    # compute the transfer curve
    point = -((-threshold / -ratio) + (-start / ratio) + -threshold)

    # apply some makeup gain
    makeup = torch.rand([1]).numpy()[0] * 6

    effects = [
        [
            "compand",
            f"{attack},{release}",
            f"{knee}:{point},{start},{threshold},{threshold}",
            f"{makeup}",
            f"{start}",
        ]
    ]

    for idx, x in enumerate(xs):
        # if the input is clipping normalize it
        if x.abs().max() >= 1.0:
            x /= x.abs().max()
            gain_db = -((torch.rand(1) * 24)).numpy().squeeze()
            x *= 10 ** (gain_db / 20.0)

        y, sr = torchaudio.sox_effects.apply_effects_tensor(
            x.view(1, -1), sr, effects, channels_first=True
        )
        xs[idx] = y

    return xs


def dynamic_range_compression(xs, sr=44100):
    """Apply a compressor."""

    attack = (torch.rand([1]).numpy()[0] * 0.05) + 0.0005
    release = (torch.rand([1]).numpy()[0] * 0.2) + attack
    knee = (torch.rand([1]).numpy()[0] * 12) + 0.0

    # design the compressor transfer function
    start = -100.0
    threshold = -((torch.rand([1]).numpy()[0] * 52) + 12)
    # threshold from -64 to -12 dB
    ratio = (torch.rand([1]).numpy()[0] * 10.0) + 1  # ratio from 1:1 to 10:1

    # compute the transfer curve
    point = threshold * (1 - (1 / ratio))

    # apply some makeup gain
    makeup = torch.rand([1]).numpy()[0] * 6

    effects = [
        [
            "compand",
            f"{attack},{release}",
            f"{knee}:{start},{threshold},{threshold},0,{point}",
            f"{makeup}",
            f"{start}",
            f"{attack}",
        ]
    ]

    for idx, x in enumerate(xs):
        y, sr = torchaudio.sox_effects.apply_effects_tensor(
            x.view(1, -1), sr, effects, channels_first=True
        )
        xs[idx] = y

    return xs


def lowpass_filter(xs, sr=44100, frequency=4000):
    effects = [["lowpass", f"{frequency}"]]

    for idx, x in enumerate(xs):
        y, sr = torchaudio.sox_effects.apply_effects_tensor(
            x, sr, effects, channels_first=True
        )
        xs[idx] = y

    return xs


def apply(xs, sr, augmentations):

    # iterate over augmentation dict
    for aug, params in augmentations.items():
        if aug == "gain":
            xs = gain(xs, **params)
        elif aug == "peak":
            xs = peaking_filter(xs, **params)
        elif aug == "lowpass":
            xs = lowpass_filter(xs, **params)
        elif aug == "pitch":
            xs = pitch_shift(xs, **params)
        elif aug == "tempo":
            xs = time_stretch(xs, **params)
        elif aug == "freq_corrupt":
            xs = frequency_corruption(xs, **params)
        else:
            raise RuntimeError("Invalid augmentation: {aug}")

    return xs
