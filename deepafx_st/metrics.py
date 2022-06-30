import torch
import auraloss
import resampy
import torchaudio
from pesq import pesq
import pyloudnorm as pyln


def crest_factor(x):
    """Compute the crest factor of waveform."""

    peak, _ = x.abs().max(dim=-1)
    rms = torch.sqrt((x ** 2).mean(dim=-1))

    return 20 * torch.log(peak / rms.clamp(1e-8))


def rms_energy(x):

    rms = torch.sqrt((x ** 2).mean(dim=-1))

    return 20 * torch.log(rms.clamp(1e-8))


def spectral_centroid(x):
    """Compute the crest factor of waveform.

    See: https://gist.github.com/endolith/359724

    """

    spectrum = torch.fft.rfft(x).abs()
    normalized_spectrum = spectrum / spectrum.sum()
    normalized_frequencies = torch.linspace(0, 1, spectrum.shape[-1])
    spectral_centroid = torch.sum(normalized_frequencies * normalized_spectrum)

    return spectral_centroid


def loudness(x, sample_rate):
    """Compute the loudness in dB LUFS of waveform."""
    meter = pyln.Meter(sample_rate)

    # add stereo dim if needed
    if x.shape[0] < 2:
        x = x.repeat(2, 1)

    return torch.tensor(meter.integrated_loudness(x.permute(1, 0).numpy()))


class MelSpectralDistance(torch.nn.Module):
    def __init__(self, sample_rate, length=65536):
        super().__init__()
        self.error = auraloss.freq.MelSTFTLoss(
            sample_rate,
            fft_size=length,
            hop_size=length,
            win_length=length,
            w_sc=0,
            w_log_mag=1,
            w_lin_mag=1,
            n_mels=128,
            scale_invariance=False,
        )

        # I think scale invariance may not work well,
        # since aspects of the phase may be considered?

    def forward(self, input, target):
        return self.error(input, target)


class PESQ(torch.nn.Module):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(self, input, target):
        if self.sample_rate != 16000:
            target = resampy.resample(
                target.view(-1).numpy(),
                self.sample_rate,
                16000,
            )
            input = resampy.resample(
                input.view(-1).numpy(),
                self.sample_rate,
                16000,
            )

        return pesq(
            16000,
            target,
            input,
            "wb",
        )


class CrestFactorError(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.nn.functional.l1_loss(
            crest_factor(input),
            crest_factor(target),
        ).item()


class RMSEnergyError(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.nn.functional.l1_loss(
            rms_energy(input),
            rms_energy(target),
        ).item()


class SpectralCentroidError(torch.nn.Module):
    def __init__(self, sample_rate, n_fft=2048, hop_length=512):
        super().__init__()

        self.spectral_centroid = torchaudio.transforms.SpectralCentroid(
            sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def forward(self, input, target):
        return torch.nn.functional.l1_loss(
            self.spectral_centroid(input + 1e-16).mean(),
            self.spectral_centroid(target + 1e-16).mean(),
        ).item()


class LoudnessError(torch.nn.Module):
    def __init__(self, sample_rate: int, peak_normalize: bool = False):
        super().__init__()
        self.sample_rate = sample_rate
        self.peak_normalize = peak_normalize

    def forward(self, input, target):

        if self.peak_normalize:
            # peak normalize
            x = input / input.abs().max()
            y = target / target.abs().max()
        else:
            x = input
            y = target

        return torch.nn.functional.l1_loss(
            loudness(x.view(1, -1), self.sample_rate),
            loudness(y.view(1, -1), self.sample_rate),
        ).item()
