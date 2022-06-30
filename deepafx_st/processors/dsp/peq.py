import torch
import numpy as np
import scipy.signal
from numba import jit

from deepafx_st.processors.processor import Processor


@jit(nopython=True)
def biqaud(
    gain_dB: float,
    cutoff_freq: float,
    q_factor: float,
    sample_rate: float,
    filter_type: str,
):
    """Use design parameters to generate coeffieicnets for a specific filter type.

    Args:
        gain_dB (float): Shelving filter gain in dB.
        cutoff_freq (float): Cutoff frequency in Hz.
        q_factor (float): Q factor.
        sample_rate (float): Sample rate in Hz.
        filter_type (str): Filter type.
            One of "low_shelf", "high_shelf", or "peaking"

    Returns:
        b (np.ndarray): Numerator filter coefficients stored as [b0, b1, b2]
        a (np.ndarray): Denominator filter coefficients stored as [a0, a1, a2]
    """

    A = 10 ** (gain_dB / 40.0)
    w0 = 2.0 * np.pi * (cutoff_freq / sample_rate)
    alpha = np.sin(w0) / (2.0 * q_factor)

    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A
    else:
        pass
        # raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    return b, a


# Adapted from https://github.com/csteinmetz1/pyloudnorm/blob/master/pyloudnorm/iirfilter.py
def parametric_eq(
    x: np.ndarray,
    sample_rate: float,
    low_shelf_gain_dB: float = 0.0,
    low_shelf_cutoff_freq: float = 80.0,
    low_shelf_q_factor: float = 0.707,
    first_band_gain_dB: float = 0.0,
    first_band_cutoff_freq: float = 300.0,
    first_band_q_factor: float = 0.707,
    second_band_gain_dB: float = 0.0,
    second_band_cutoff_freq: float = 1000.0,
    second_band_q_factor: float = 0.707,
    third_band_gain_dB: float = 0.0,
    third_band_cutoff_freq: float = 4000.0,
    third_band_q_factor: float = 0.707,
    fourth_band_gain_dB: float = 0.0,
    fourth_band_cutoff_freq: float = 8000.0,
    fourth_band_q_factor: float = 0.707,
    high_shelf_gain_dB: float = 0.0,
    high_shelf_cutoff_freq: float = 1000.0,
    high_shelf_q_factor: float = 0.707,
    dtype=np.float32,
):
    """Six-band parametric EQ.

    Low-shelf -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf

    Args:


    """
    # print(f"autodiff peq fs = {sample_rate}")

    # -------- apply low-shelf filter --------
    b, a = biqaud(
        low_shelf_gain_dB,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    sos0 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    # -------- apply first-band peaking filter --------
    b, a = biqaud(
        first_band_gain_dB,
        first_band_cutoff_freq,
        first_band_q_factor,
        sample_rate,
        "peaking",
    )
    sos1 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    # -------- apply second-band peaking filter --------
    b, a = biqaud(
        second_band_gain_dB,
        second_band_cutoff_freq,
        second_band_q_factor,
        sample_rate,
        "peaking",
    )
    sos2 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    # -------- apply third-band peaking filter --------
    b, a = biqaud(
        third_band_gain_dB,
        third_band_cutoff_freq,
        third_band_q_factor,
        sample_rate,
        "peaking",
    )
    sos3 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    # -------- apply fourth-band peaking filter --------
    b, a = biqaud(
        fourth_band_gain_dB,
        fourth_band_cutoff_freq,
        fourth_band_q_factor,
        sample_rate,
        "peaking",
    )
    sos4 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    # -------- apply high-shelf filter --------
    b, a = biqaud(
        high_shelf_gain_dB,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    sos5 = np.concatenate((b, a))
    x = scipy.signal.lfilter(b, a, x)

    return x.astype(dtype)


class ParametricEQ(Processor):
    def __init__(
        self,
        sample_rate,
        min_gain_dB=-24.0,
        default_gain_dB=0.0,
        max_gain_dB=24.0,
        min_q_factor=0.1,
        default_q_factor=0.707,
        max_q_factor=10,
        eps=1e-8,
    ):
        """ """
        super().__init__()
        self.sample_rate = sample_rate
        self.eps = eps
        self.ports = [
            {
                "name": "Lowshelf gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Lowshelf cutoff",
                "min": 20.0,
                "max": 200.0,
                "default": 100.0,
                "units": "Hz",
            },
            {
                "name": "Lowshelf Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
            {
                "name": "First band gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "First band cutoff",
                "min": 200.0,
                "max": 2000.0,
                "default": 400.0,
                "units": "Hz",
            },
            {
                "name": "First band Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": 0.707,
                "units": "",
            },
            {
                "name": "Second band gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Second band cutoff",
                "min": 800.0,
                "max": 4000.0,
                "default": 1000.0,
                "units": "Hz",
            },
            {
                "name": "Second band Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
            {
                "name": "Third band gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Third band cutoff",
                "min": 2000.0,
                "max": 8000.0,
                "default": 4000.0,
                "units": "Hz",
            },
            {
                "name": "Third band Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
            {
                "name": "Fourth band gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Fourth band cutoff",
                "min": 4000.0,
                "max": (24000 // 2) * 0.9,
                "default": 8000.0,
                "units": "Hz",
            },
            {
                "name": "Fourth band Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
            {
                "name": "Highshelf gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Highshelf cutoff",
                "min": 4000.0,
                "max": (24000 // 2) * 0.9,
                "default": 8000.0,
                "units": "Hz",
            },
            {
                "name": "Highshelf Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
        ]

        self.num_control_params = len(self.ports)
        self.process_fn = parametric_eq

    def forward(self, x, p, sample_rate=24000, **kwargs):
        "All processing in the forward is in numpy."
        return self.run_series(x, p, sample_rate)
