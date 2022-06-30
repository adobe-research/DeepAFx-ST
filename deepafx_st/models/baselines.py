import torch
import torchaudio
import scipy.signal
import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
from deepafx_st.processors.dsp.compressor import compressor

from tqdm import tqdm


class BaselineEQ(torch.nn.Module):
    def __init__(
        self,
        ntaps: int = 63,
        n_fft: int = 65536,
        sample_rate: float = 44100,
    ):
        super().__init__()
        self.ntaps = ntaps
        self.n_fft = n_fft
        self.sample_rate = sample_rate

        # compute the target spectrum
        # print("Computing target spectrum...")
        # self.target_spec, self.sm_target_spec = self.analyze_speech_dataset(filepaths)
        # self.plot_spectrum(self.target_spec, filename="targetEQ")
        # self.plot_spectrum(self.sm_target_spec, filename="targetEQsm")

    def forward(self, x, y):

        bs, ch, s = x.size()

        x = x.view(bs * ch, -1)
        y = y.view(bs * ch, -1)

        in_spec = self.get_average_spectrum(x)
        ref_spec = self.get_average_spectrum(y)

        sm_in_spec = self.smooth_spectrum(in_spec)
        sm_ref_spec = self.smooth_spectrum(ref_spec)

        # self.plot_spectrum(in_spec, filename="inSpec")
        # self.plot_spectrum(sm_in_spec, filename="inSpecsm")

        # design inverse FIR filter to match target EQ
        freqs = np.linspace(0, 1.0, num=(self.n_fft // 2) + 1)
        response = sm_ref_spec / sm_in_spec
        response[-1] = 0.0  # zero gain at nyquist

        b = scipy.signal.firwin2(
            self.ntaps,
            freqs * (self.sample_rate / 2),
            response,
            fs=self.sample_rate,
        )

        # scale the coefficients for less intense filter
        # clearb *= 0.5

        # apply the filter
        x_filt = scipy.signal.lfilter(b, [1.0], x.numpy())
        x_filt = torch.tensor(x_filt.astype("float32"))

        if False:
            # plot the filter response
            w, h = scipy.signal.freqz(b, fs=self.sample_rate, worN=response.shape[-1])

            fig, ax1 = plt.subplots()
            ax1.set_title("Digital filter frequency response")
            ax1.plot(w, 20 * np.log10(abs(h + 1e-8)))
            ax1.plot(w, 20 * np.log10(abs(response + 1e-8)))

            ax1.set_xscale("log")
            ax1.set_ylim([-12, 12])
            plt.grid(c="lightgray")
            plt.savefig(f"inverse.png")

            x_filt_avg_spec = self.get_average_spectrum(x_filt)
            sm_x_filt_avg_spec = self.smooth_spectrum(x_filt_avg_spec)
            y_avg_spec = self.get_average_spectrum(y)
            sm_y_avg_spec = self.smooth_spectrum(y_avg_spec)
            compare = torch.stack(
                [
                    torch.tensor(sm_in_spec),
                    torch.tensor(sm_x_filt_avg_spec),
                    torch.tensor(sm_ref_spec),
                    torch.tensor(sm_y_avg_spec),
                ]
            )
            self.plot_multi_spectrum(
                compare,
                legend=["in", "out", "target curve", "actual target"],
                filename="outSpec",
            )

        return x_filt

    def analyze_speech_dataset(self, filepaths, peak=-3.0):
        avg_spec = []
        for filepath in tqdm(filepaths, ncols=80):
            x, sr = torchaudio.load(filepath)
            x /= x.abs().max()
            x *= 10 ** (peak / 20.0)
            avg_spec.append(self.get_average_spectrum(x))
        avg_specs = torch.stack(avg_spec)

        avg_spec = avg_specs.mean(dim=0).numpy()
        avg_spec_std = avg_specs.std(dim=0).numpy()

        # self.plot_multi_spectrum(avg_specs, filename="allTargetEQs")
        # self.plot_spectrum_stats(avg_spec, avg_spec_std, filename="targetEQstats")

        sm_avg_spec = self.smooth_spectrum(avg_spec)

        return avg_spec, sm_avg_spec

    def smooth_spectrum(self, H):
        # apply Savgol filter for smoothed target curve
        return scipy.signal.savgol_filter(H, 1025, 2)

    def get_average_spectrum(self, x):

        # x = x[:, : self.n_fft]
        X = torch.stft(x, self.n_fft, return_complex=True, normalized=True)
        # fft_size = self.next_power_of_2(x.shape[-1])
        # X = torch.fft.rfft(x, n=fft_size)

        X = X.abs()  # convert to magnitude
        X = X.mean(dim=-1).view(-1)  # average across frames

        return X

    @staticmethod
    def next_power_of_2(x):
        return 1 if x == 0 else int(2 ** np.ceil(np.log2(x)))

    def plot_multi_spectrum(self, Hs, legend=[], filename=None):

        bin_width = (self.sample_rate / 2) / (self.n_fft // 2)
        freqs = np.arange(0, (self.sample_rate / 2) + bin_width, step=bin_width)

        fig, ax1 = plt.subplots()

        for H in Hs:
            ax1.plot(
                freqs,
                20 * np.log10(abs(H) + 1e-8),
            )

        plt.legend(legend)

        # avg_spec = Hs.mean(dim=0).numpy()
        # ax1.plot(freqs, 20 * np.log10(avg_spec), color="k", linewidth=2)

        ax1.set_xscale("log")
        ax1.set_ylim([-80, 0])
        plt.grid(c="lightgray")

        if filename is not None:
            plt.savefig(f"{filename}.png")

    def plot_spectrum_stats(self, H_mean, H_std, filename=None):
        bin_width = (self.sample_rate / 2) / (self.n_fft // 2)
        freqs = np.arange(0, (self.sample_rate / 2) + bin_width, step=bin_width)

        fig, ax1 = plt.subplots()
        ax1.plot(freqs, 20 * np.log10(H_mean))
        ax1.plot(
            freqs,
            (20 * np.log10(H_mean)) + (20 * np.log10(H_std)),
            linestyle="--",
            color="k",
        )
        ax1.plot(
            freqs,
            (20 * np.log10(H_mean)) - (20 * np.log10(H_std)),
            linestyle="--",
            color="k",
        )

        ax1.set_xscale("log")
        ax1.set_ylim([-80, 0])
        plt.grid(c="lightgray")

        if filename is not None:
            plt.savefig(f"{filename}.png")

    def plot_spectrum(self, H, legend=[], filename=None):

        bin_width = (self.sample_rate / 2) / (self.n_fft // 2)
        freqs = np.arange(0, (self.sample_rate / 2) + bin_width, step=bin_width)

        fig, ax1 = plt.subplots()
        ax1.plot(freqs, 20 * np.log10(H))
        ax1.set_xscale("log")
        ax1.set_ylim([-80, 0])
        plt.grid(c="lightgray")

        plt.legend(legend)

        if filename is not None:
            plt.savefig(f"{filename}.png")


class BaslineComp(torch.nn.Module):
    def __init__(
        self,
        sample_rate: float = 44100,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)

    def forward(self, x, y):

        x_lufs = self.meter.integrated_loudness(x.view(-1).numpy())
        y_lufs = self.meter.integrated_loudness(y.view(-1).numpy())

        delta_lufs = y_lufs - x_lufs

        threshold = 0.0
        x_comp = x
        x_comp_new = x
        while delta_lufs > 0.5 and threshold > -80.0:
            x_comp = x_comp_new  # use the last setting
            x_comp_new = compressor(
                x.view(-1).numpy(),
                self.sample_rate,
                threshold=threshold,
                ratio=3,
                attack_time=0.001,
                release_time=0.05,
                knee_dB=6.0,
                makeup_gain_dB=0.0,
            )
            x_comp_new = torch.tensor(x_comp_new)
            x_comp_new /= x_comp_new.abs().max()
            x_comp_new *= 10 ** (-12.0 / 20)
            x_lufs = self.meter.integrated_loudness(x_comp_new.view(-1).numpy())
            delta_lufs = y_lufs - x_lufs
            threshold -= 0.5

        return x_comp.view(1, 1, -1)


class BaselineEQAndComp(torch.nn.Module):
    def __init__(
        self,
        ntaps=63,
        n_fft=65536,
        sample_rate=44100,
        block_size=1024,
        plugin_config=None,
    ):
        super().__init__()
        self.eq = BaselineEQ(ntaps, n_fft, sample_rate)
        self.comp = BaslineComp(sample_rate)

    def forward(self, x, y):

        with torch.inference_mode():
            x /= x.abs().max()
            y /= y.abs().max()
            x *= 10 ** (-12.0 / 20)
            y *= 10 ** (-12.0 / 20)

            x = self.eq(x, y)

            x /= x.abs().max()
            y /= y.abs().max()
            x *= 10 ** (-12.0 / 20)
            y *= 10 ** (-12.0 / 20)

            x = self.comp(x, y)

            x /= x.abs().max()
            x *= 10 ** (-12.0 / 20)

        return x
