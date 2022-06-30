import math
import torch
import scipy.signal

import deepafx_st.processors.autodiff.signal
from deepafx_st.processors.processor import Processor


@torch.jit.script
def compressor(
    x: torch.Tensor,
    sample_rate: float,
    threshold: torch.Tensor,
    ratio: torch.Tensor,
    attack_time: torch.Tensor,
    release_time: torch.Tensor,
    knee_dB: torch.Tensor,
    makeup_gain_dB: torch.Tensor,
    eps: float = 1e-8,
):
    """Note the `release` parameter is not used."""
    # print(f"autodiff comp fs = {sample_rate}")

    s = x.size()  # should be one 1d

    threshold = threshold.squeeze()
    ratio = ratio.squeeze()
    attack_time = attack_time.squeeze()
    makeup_gain_dB = makeup_gain_dB.squeeze()

    # uni-polar dB signal
    # Turn the input signal into a uni-polar signal on the dB scale
    x_G = 20 * torch.log10(torch.abs(x) + 1e-8)  # x_uni casts type

    # Ensure there are no values of negative infinity
    x_G = torch.clamp(x_G, min=-96)

    # Static characteristics with knee
    y_G = torch.zeros(s).type_as(x)

    ratio = ratio.view(-1)
    threshold = threshold.view(-1)
    attack_time = attack_time.view(-1)
    release_time = release_time.view(-1)
    knee_dB = knee_dB.view(-1)
    makeup_gain_dB = makeup_gain_dB.view(-1)

    # Below knee
    idx = torch.where((2 * (x_G - threshold)) < -knee_dB)[0]
    y_G[idx] = x_G[idx]

    # At knee
    idx = torch.where((2 * torch.abs(x_G - threshold)) <= knee_dB)[0]
    y_G[idx] = x_G[idx] + (
        (1 / ratio) * (((x_G[idx] - threshold + knee_dB) / 2) ** 2)
    ) / (2 * knee_dB)

    # Above knee threshold
    idx = torch.where((2 * (x_G - threshold)) > knee_dB)[0]
    y_G[idx] = threshold + ((x_G[idx] - threshold) / ratio)

    x_L = x_G - y_G

    # design 1-pole butterworth lowpass
    fc = 1.0 / (attack_time * sample_rate)
    b, a = deepafx_st.processors.autodiff.signal.butter(fc)

    # apply FIR approx of IIR filter
    y_L = deepafx_st.processors.autodiff.signal.approx_iir_filter(b, a, x_L)

    lin_y_L = torch.pow(10.0, -y_L / 20.0)  # convert back to linear
    y = lin_y_L * x  # apply gain

    # apply makeup gain
    y *= torch.pow(10.0, makeup_gain_dB / 20.0)

    return y


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
                "units": "dB",
            },
            {
                "name": "Ratio",
                "min": min_ratio,
                "max": max_ratio,
                "default": 2.0,
                "units": "",
            },
            {
                "name": "Attack",
                "min": min_attack,
                "max": max_attack,
                "default": 0.001,
                "units": "s",
            },
            {
                # this is a dummy parameter
                "name": "Release (dummy)",
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

    def forward(self, x, p, sample_rate=24000, **kwargs):
        """

        Assume that parameters in p are normalized between 0 and 1.

        x (tensor): Shape batch x 1 x samples
        p (tensor): shape batch x params

        """
        bs, ch, s = x.size()

        inputs = torch.split(x, 1, 0)
        params = torch.split(p, 1, 0)

        y = []  # loop over batch dimension
        for input, param in zip(inputs, params):
            denorm_param = self.denormalize_params(param.view(-1))
            y.append(compressor(input.view(-1), sample_rate, *denorm_param))

        return torch.stack(y, dim=0).view(bs, 1, -1)
