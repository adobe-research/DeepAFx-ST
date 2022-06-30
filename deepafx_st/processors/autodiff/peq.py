import torch

import deepafx_st.processors.autodiff.signal
from deepafx_st.processors.processor import Processor


@torch.jit.script
def parametric_eq(
    x: torch.Tensor,
    sample_rate: float,
    low_shelf_gain_dB: torch.Tensor,
    low_shelf_cutoff_freq: torch.Tensor,
    low_shelf_q_factor: torch.Tensor,
    first_band_gain_dB: torch.Tensor,
    first_band_cutoff_freq: torch.Tensor,
    first_band_q_factor: torch.Tensor,
    second_band_gain_dB: torch.Tensor,
    second_band_cutoff_freq: torch.Tensor,
    second_band_q_factor: torch.Tensor,
    third_band_gain_dB: torch.Tensor,
    third_band_cutoff_freq: torch.Tensor,
    third_band_q_factor: torch.Tensor,
    fourth_band_gain_dB: torch.Tensor,
    fourth_band_cutoff_freq: torch.Tensor,
    fourth_band_q_factor: torch.Tensor,
    high_shelf_gain_dB: torch.Tensor,
    high_shelf_cutoff_freq: torch.Tensor,
    high_shelf_q_factor: torch.Tensor,
):
    """Six-band parametric EQ.

    Low-shelf -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf

    Args:
        x (torch.Tensor): 1d signal.


    """
    a_s, b_s = [], []
    #print(f"autodiff peq fs = {sample_rate}")

    # -------- apply low-shelf filter --------
    b, a = deepafx_st.processors.autodiff.signal.biqaud(
        low_shelf_gain_dB,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply first-band peaking filter --------
    b, a = deepafx_st.processors.autodiff.signal.biqaud(
        first_band_gain_dB,
        first_band_cutoff_freq,
        first_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply second-band peaking filter --------
    b, a = deepafx_st.processors.autodiff.signal.biqaud(
        second_band_gain_dB,
        second_band_cutoff_freq,
        second_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply third-band peaking filter --------
    b, a = deepafx_st.processors.autodiff.signal.biqaud(
        third_band_gain_dB,
        third_band_cutoff_freq,
        third_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply fourth-band peaking filter --------
    b, a = deepafx_st.processors.autodiff.signal.biqaud(
        fourth_band_gain_dB,
        fourth_band_cutoff_freq,
        fourth_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply high-shelf filter --------
    b, a = deepafx_st.processors.autodiff.signal.biqaud(
        high_shelf_gain_dB,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    b_s.append(b)
    a_s.append(a)

    x = deepafx_st.processors.autodiff.signal.approx_iir_filter_cascade(
        b_s, a_s, x.view(-1)
    )

    return x


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
                "min": 200.0,
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

    def forward(self, x, p, sample_rate=24000, **kwargs):

        bs, chs, s = x.size()

        inputs = torch.split(x, 1, 0)
        params = torch.split(p, 1, 0)

        y = []  # loop over batch dimension
        for input, param in zip(inputs, params):
            denorm_param = self.denormalize_params(param.view(-1))
            y.append(parametric_eq(input.view(-1), sample_rate, *denorm_param))

        return torch.stack(y, dim=0).view(bs, 1, -1)
