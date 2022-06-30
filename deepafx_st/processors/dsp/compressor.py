import sys
import torch
import numpy as np
import scipy.signal
from numba import jit

from deepafx_st.processors.processor import Processor


# Adapted from: https://github.com/drscotthawley/signaltrain/blob/master/signaltrain/audio.py
@jit(nopython=True)
def my_clip_min(
    x: np.ndarray,
    clip_min: float,
):  # does the work of np.clip(), which numba doesn't support yet
    # TODO: keep an eye on Numba PR https://github.com/numba/numba/pull/3468 that fixes this
    inds = np.where(x < clip_min)
    x[inds] = clip_min
    return x


@jit(nopython=True)
def compressor(
    x: np.ndarray,
    sample_rate: float,
    threshold: float = -24.0,
    ratio: float = 2.0,
    attack_time: float = 0.01,
    release_time: float = 0.01,
    knee_dB: float = 0.0,
    makeup_gain_dB: float = 0.0,
    dtype=np.float32,
):
    """

    Args:
        x (np.ndarray): Input signal.
        sample_rate (float): Sample rate in Hz.
        threshold (float): Threhold in dB.
        ratio (float): Ratio (should be >=1 , i.e. ratio:1).
        attack_time (float): Attack time in seconds.
        release_time (float): Release time in seconds.
        knee_dB (float): Knee.
        makeup_gain_dB (float): Makeup Gain.
        dtype (type): Output type. Default: np.float32

    Returns:
        y (np.ndarray): Output signal.

    """
    # print(f"dsp comp fs = {sample_rate}")

    N = len(x)
    dtype = x.dtype
    y = np.zeros(N, dtype=dtype)

    # Initialize separate attack and release times
    # Where do these numbers come from
    alpha_A = np.exp(-np.log(9) / (sample_rate * attack_time))
    alpha_R = np.exp(-np.log(9) / (sample_rate * release_time))

    # Turn the input signal into a uni-polar signal on the dB scale
    x_G = 20 * np.log10(np.abs(x) + 1e-8)  # x_uni casts type

    # Ensure there are no values of negative infinity
    x_G = my_clip_min(x_G, -96)

    # Static characteristics with knee
    y_G = np.zeros(N, dtype=dtype)

    # Below knee
    idx = np.where((2 * (x_G - threshold)) < -knee_dB)
    y_G[idx] = x_G[idx]

    # At knee
    idx = np.where((2 * np.abs(x_G - threshold)) <= knee_dB)
    y_G[idx] = x_G[idx] + (
        (1 / ratio) * (((x_G[idx] - threshold + knee_dB) / 2) ** 2)
    ) / (2 * knee_dB)

    # Above knee threshold
    idx = np.where((2 * (x_G - threshold)) > knee_dB)
    y_G[idx] = threshold + ((x_G[idx] - threshold) / ratio)

    x_L = x_G - y_G

    # this loop is slow but not vectorizable due to its cumulative, sequential nature. @autojit makes it fast(er).
    y_L = np.zeros(N, dtype=dtype)
    for n in range(1, N):
        # smooth over the gainChange
        if x_L[n] > y_L[n - 1]:  # attack mode
            y_L[n] = (alpha_A * y_L[n - 1]) + ((1 - alpha_A) * x_L[n])
        else:  # release
            y_L[n] = (alpha_R * y_L[n - 1]) + ((1 - alpha_R) * x_L[n])

    # Convert to linear amplitude scalar; i.e. map from dB to amplitude
    lin_y_L = np.power(10.0, (-y_L / 20.0))
    y = lin_y_L * x  # Apply linear amplitude to input sample

    y *= np.power(10.0, makeup_gain_dB / 20.0)  # apply makeup gain

    return y.astype(dtype)


class Compressor(Processor):
    def __init__(
        self,
        sample_rate,
        max_threshold=0.0,
        min_threshold=-80,
        max_ratio=20.0,
        min_ratio=1.0,
        max_attack=0.1,
        min_attack=0.0001,
        max_release=1.0,
        min_release=0.005,
        max_knee=12.0,
        min_knee=0.0,
        max_mkgain=48.0,
        min_mkgain=-48.0,
        eps=1e-8,
    ):
        """ """
        super().__init__()
        self.sample_rate = sample_rate
        self.eps = eps
        self.ports = [
            {
                "name": "Threshold",
                "min": min_threshold,
                "max": max_threshold,
                "default": -12.0,
                "units": "",
            },
            {
                "name": "Ratio",
                "min": min_ratio,
                "max": max_ratio,
                "default": 2.0,
                "units": "",
            },
            {
                "name": "Attack Time",
                "min": min_attack,
                "max": max_attack,
                "default": 0.001,
                "units": "s",
            },
            {
                "name": "Release Time",
                "min": min_release,
                "max": max_release,
                "default": 0.045,
                "units": "s",
            },
            {
                "name": "Knee",
                "min": min_knee,
                "max": max_knee,
                "default": 6.0,
                "units": "dB",
            },
            {
                "name": "Makeup Gain",
                "min": min_mkgain,
                "max": max_mkgain,
                "default": 0.0,
                "units": "dB",
            },
        ]

        self.num_control_params = len(self.ports)
        self.process_fn = compressor

    def forward(self, x, p, sample_rate=24000, **kwargs):
        "All processing in the forward is in numpy."
        return self.run_series(x, p, sample_rate)
