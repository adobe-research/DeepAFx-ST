import torch
import auraloss
import torchaudio
import numpy as np
import scipy.signal
from tqdm import tqdm
from itertools import chain
from time import perf_counter

from deepafx_st.models.encoder import SpectralEncoder
from deepafx_st.models.controller import StyleTransferController
from deepafx_st.processors.autodiff.channel import AutodiffChannel
from deepafx_st.processors.proxy.channel import ProxyChannel
from deepafx_st.processors.dsp.compressor import Compressor
from deepafx_st.processors.dsp.peq import ParametricEQ
from deepafx_st.processors.spsa.channel import SPSAChannel
from deepafx_st.utils import DSPMode, count_parameters
from deepafx_st.processors.dsp.compressor import compressor


def run_dsp(x, peq_p, comp_p, peq, comp):

    x = peq(x, peq_p)
    x = comp(x, comp_p)

    return x


if __name__ == "__main__":

    sample_rate = 24000
    n_iters = 1000
    length_sec = 5
    bs = 4
    length_samp = sample_rate * length_sec

    # loss
    mrstft_loss = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[32, 128, 512, 2048, 8192, 32768],
        hop_sizes=[16, 64, 256, 1024, 4096, 16384],
        win_lengths=[32, 128, 512, 2048, 8192, 32768],
        w_sc=0.0,
        w_phs=0.0,
        w_lin_mag=1.0,
        w_log_mag=1.0,
    )

    # dsp effects
    peq_dsp = ParametricEQ(sample_rate)
    comp_dsp = Compressor(sample_rate)

    # autodiff effects
    channel_ad = AutodiffChannel(sample_rate)

    spsa = SPSAChannel(sample_rate, True, bs)

    # proxy channel tcn 1
    np_norm = ProxyChannel(
        [],
        freeze_proxies=True,
        dsp_mode=DSPMode.NONE,
        tcn_nblocks=4,
        tcn_dilation_growth=8,
        tcn_channel_width=64,
        tcn_kernel_size=13,
        num_tcns=1,
        sample_rate=sample_rate,
    )

    # proxy channel tcn 1
    np_hh = ProxyChannel(
        [],
        freeze_proxies=True,
        dsp_mode=DSPMode.INFER,
        tcn_nblocks=4,
        tcn_dilation_growth=8,
        tcn_channel_width=64,
        tcn_kernel_size=13,
        num_tcns=1,
        sample_rate=sample_rate,
    )

    # proxy channel tcn 1
    np_fh = ProxyChannel(
        [],
        freeze_proxies=True,
        dsp_mode=DSPMode.TRAIN_INFER,
        tcn_nblocks=4,
        tcn_dilation_growth=8,
        tcn_channel_width=64,
        tcn_kernel_size=13,
        num_tcns=1,
        sample_rate=sample_rate,
    )

    # proxy channel tcn 1
    tcn1 = ProxyChannel(
        [],
        freeze_proxies=False,
        dsp_mode=DSPMode.NONE,
        tcn_nblocks=4,
        tcn_dilation_growth=8,
        tcn_channel_width=64,
        tcn_kernel_size=13,
        num_tcns=1,
        sample_rate=sample_rate,
    )

    # proxy channel tcn 2
    tcn2 = ProxyChannel(
        [],
        freeze_proxies=False,
        dsp_mode=DSPMode.NONE,
        tcn_nblocks=4,
        tcn_dilation_growth=8,
        tcn_channel_width=64,
        tcn_kernel_size=13,
        num_tcns=2,
        sample_rate=sample_rate,
    )

    # predictor models
    encoder = SpectralEncoder(
        channel_ad.num_control_params,
        sample_rate,
        encoder_model="efficient_net",
        embed_dim=1024,
        width_mult=1,
    )
    controller = StyleTransferController(
        channel_ad.num_control_params,
        1024,
        # bottleneck_dim=-1,
    )

    print()

    # iterate
    for model in [
        "rb_infer",
        "dsp_infer",
        "autodiff_cpu_infer",
        "autodiff_gpu_infer",
        "tcn1_cpu_infer",
        "tcn2_cpu_infer",
        "tcn1_gpu_infer",
        "tcn2_gpu_infer",
        "autodiff_gpu_grad",
        "np_norm_gpu_grad",
        "np_hh_gpu_grad",
        "np_fh_gpu_grad",
        "tcn1_gpu_grad",
        "tcn2_gpu_grad",
        "spsa_gpu_grad",
    ]:
        timings = []
        for n in tqdm(range(n_iters), ncols=80):
            if "grad" in model:
                eff_bs = bs
            else:
                eff_bs = 1
            if model == "rb_infer":
                if n == 0:
                    p = torch.rand(
                        eff_bs, channel_ad.num_control_params, requires_grad=True
                    )
                    x = torch.randn(eff_bs, 1, length_samp)
                    y = torch.randn(eff_bs, 1, length_samp)

                    n_fft = 65536
                    freqs = np.linspace(0, 1.0, num=(n_fft // 2) + 1)
                    response = np.random.rand(n_fft // 2 + 1)
                    response[-1] = 0.0  # zero gain at nyquist
                    b = scipy.signal.firwin2(
                        63,
                        freqs * (sample_rate / 2),
                        response,
                        fs=sample_rate,
                    )

                t1_start = perf_counter()

                x_filt = scipy.signal.lfilter(b, [1.0], x.numpy())
                x_filt = torch.tensor(x_filt.astype("float32"))

                with torch.inference_mode():
                    x_comp_new = compressor(
                        x_filt.view(-1).numpy(),
                        sample_rate,
                        threshold=-12,
                        ratio=3,
                        attack_time=0.001,
                        release_time=0.05,
                        knee_dB=6.0,
                        makeup_gain_dB=0.0,
                    )

                t1_stop = perf_counter()

            if model == "dsp_infer":
                if n == 0:
                    params = 0
                    x = np.random.rand(length_samp)
                    peq_p = np.random.rand(peq_dsp.num_control_params)
                    comp_p = np.random.rand(comp_dsp.num_control_params)
                t1_start = perf_counter()
                y = run_dsp(x, peq_p, comp_p, peq_dsp, comp_dsp)
                t1_stop = perf_counter()
            elif "autodiff" in model:
                if n == 0:
                    params = 0
                    p = torch.rand(
                        eff_bs, channel_ad.num_control_params, requires_grad=True
                    )
                    x = torch.randn(eff_bs, 1, length_samp)
                    y = torch.randn(eff_bs, 1, length_samp)
                    optimizer = torch.optim.Adam(
                        chain(
                            encoder.parameters(),
                            controller.parameters(),
                        ),
                        lr=1e-3,
                    )

                    if "gpu" in model:
                        p = p.to("cuda")
                        x = x.to("cuda")
                        y = y.to("cuda")
                        if "grad" in model:
                            encoder.to("cuda")
                            controller.to("cuda")

                if "grad" in model:
                    t1_start = perf_counter()
                    e_x = encoder(x)
                    e_y = encoder(y)
                    p = controller(e_x, e_y)
                    y_hat = channel_ad(x, p)
                    loss = mrstft_loss(y_hat, x)
                    loss.backward()
                    optimizer.step()
                    t1_stop = perf_counter()
                else:
                    with torch.inference_mode():
                        t1_start = perf_counter()
                        y = channel_ad(x, p)
                        t1_stop = perf_counter()

            elif "tcn1" in model:
                if n == 0:
                    params = count_parameters(tcn1)
                    p = torch.rand(
                        eff_bs,
                        channel_ad.num_control_params,
                        requires_grad=False,
                    )
                    x = torch.randn(eff_bs, 1, length_samp)
                    y = torch.randn(eff_bs, 1, length_samp)
                    optimizer = torch.optim.Adam(
                        chain(
                            encoder.parameters(),
                            controller.parameters(),
                            tcn1.parameters(),
                        ),
                        lr=1e-3,
                    )

                    if "gpu" in model:
                        p = p.to("cuda")
                        x = x.to("cuda")
                        y = y.to("cuda")
                        tcn1.to("cuda")

                        if "grad" in model:
                            encoder.to("cuda")
                            controller.to("cuda")
                    else:
                        tcn1.to("cpu")

                if "grad" in model:
                    t1_start = perf_counter()
                    e_x = encoder(x)
                    e_y = encoder(y)
                    p = controller(e_x, e_y)
                    y_hat = tcn1(x, p)
                    loss = mrstft_loss(y_hat, x)
                    loss.backward()
                    optimizer.step()
                    t1_stop = perf_counter()
                else:
                    with torch.inference_mode():
                        t1_start = perf_counter()
                        y = tcn1(x, p)
                        t1_stop = perf_counter()

            elif "tcn2" in model:
                if n == 0:
                    params = count_parameters(tcn2)
                    p = torch.rand(
                        eff_bs, channel_ad.num_control_params, requires_grad=True
                    )
                    x = torch.randn(eff_bs, 1, length_samp)
                    y = torch.randn(eff_bs, 1, length_samp)
                    optimizer = torch.optim.Adam(
                        chain(
                            encoder.parameters(),
                            controller.parameters(),
                            tcn2.parameters(),
                        ),
                        lr=1e-3,
                    )

                    if "gpu" in model:
                        p = p.to("cuda")
                        x = x.to("cuda")
                        y = y.to("cuda")
                        tcn2.to("cuda")

                        if "grad" in model:
                            encoder.to("cuda")
                            controller.to("cuda")

                if "grad" in model:
                    t1_start = perf_counter()
                    e_x = encoder(x)
                    e_y = encoder(y)
                    p = controller(e_x, e_y)
                    y_hat = tcn2(x, p)
                    loss = mrstft_loss(y_hat, x)
                    loss.backward()
                    optimizer.step()
                    t1_stop = perf_counter()
                else:
                    with torch.inference_mode():
                        t1_start = perf_counter()
                        y = tcn2(x, p)
                        t1_stop = perf_counter()
            elif "np" in model:
                if n == 0:
                    p = torch.rand(
                        eff_bs, channel_ad.num_control_params, requires_grad=True
                    )
                    x = torch.randn(eff_bs, 1, length_samp)
                    y = torch.randn(eff_bs, 1, length_samp)
                    optimizer = torch.optim.Adam(
                        chain(
                            encoder.parameters(),
                            controller.parameters(),
                        ),
                        lr=1e-3,
                    )

                    if "gpu" in model:
                        p = p.to("cuda")
                        x = x.to("cuda")
                        y = y.to("cuda")
                        if "grad" in model:
                            encoder.to("cuda")
                            controller.to("cuda")
                            np_norm.to("cuda")
                            np_fh.to("cuda")
                            np_hh.to("cuda")

                if "grad" in model:
                    t1_start = perf_counter()
                    e_x = encoder(x)
                    e_y = encoder(y)
                    p = controller(e_x, e_y)

                    if "fh" in model:
                        y_hat = np_fh(x, p)
                    elif "hh" in model:
                        y_hat = np_hh(x, p)
                    else:
                        y_hat = np_norm(x, p)

                    loss = mrstft_loss(y_hat, x)
                    loss.backward()
                    optimizer.step()
                    t1_stop = perf_counter()

            elif "spsa" in model:
                if n == 0:
                    p = torch.rand(
                        eff_bs, channel_ad.num_control_params, requires_grad=True
                    )
                    x = torch.randn(eff_bs, 1, length_samp)
                    y = torch.randn(eff_bs, 1, length_samp)
                    optimizer = torch.optim.Adam(
                        chain(
                            encoder.parameters(),
                            controller.parameters(),
                        ),
                        lr=1e-3,
                    )
                    if "gpu" in model:
                        p = p.to("cuda")
                        x = x.to("cuda")
                        y = y.to("cuda")
                        if "grad" in model:
                            encoder.to("cuda")
                            controller.to("cuda")
                            spsa.to("cuda")

                if "grad" in model:
                    t1_start = perf_counter()
                    e_x = encoder(x)
                    e_y = encoder(y)
                    p = controller(e_x, e_y)
                    y_hat = spsa(x, p)
                    loss = mrstft_loss(y_hat, x)
                    loss.backward()
                    optimizer.step()
                    t1_stop = perf_counter()

            elapsed = t1_stop - t1_start
            timings.append(elapsed)

        # remove the first time
        timings = timings[10:]

        rtf = np.mean(timings) / length_sec
        sec_per_step = np.mean(timings)
        print(f"{model} : sec/step {sec_per_step:0.4f}    {rtf:0.4f} RTF")
