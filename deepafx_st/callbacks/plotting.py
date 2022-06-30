import io
import torch
import PIL.Image
import numpy as np
import scipy.signal
import librosa.display
import matplotlib.pyplot as plt

from torch.functional import Tensor
from torchvision.transforms import ToTensor


def compute_comparison_spectrogram(
    x: np.ndarray,
    y: np.ndarray,
    sample_rate: float = 44100,
    n_fft: int = 2048,
    hop_length: int = 1024,
) -> Tensor:
    X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
    X_db = librosa.amplitude_to_db(np.abs(X), ref=np.max)

    Y = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    Y_db = librosa.amplitude_to_db(np.abs(Y), ref=np.max)

    fig, axs = plt.subplots(figsize=(9, 6), nrows=2)
    img = librosa.display.specshow(
        X_db,
        ax=axs[0],
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        sr=sample_rate,
    )
    # fig.colorbar(img, ax=axs[0])
    img = librosa.display.specshow(
        Y_db,
        ax=axs[1],
        hop_length=hop_length,
        x_axis="time",
        y_axis="log",
        sr=sample_rate,
    )
    # fig.colorbar(img, ax=axs[1])

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close("all")

    return image


def plot_multi_spectrum(
    ys=None,
    Hs=None,
    legend=[],
    title="Spectrum",
    filename=None,
    sample_rate=44100,
    n_fft=1024,
    zero_mean=False,
):

    if Hs is None:
        Hs = []
        for y in ys:
            X = get_average_spectrum(y, n_fft)
            X_sm = smooth_spectrum(X)
            Hs.append(X_sm)

    bin_width = (sample_rate / 2) / (n_fft // 2)
    freqs = np.arange(0, (sample_rate / 2) + bin_width, step=bin_width)

    fig, ax1 = plt.subplots()

    for idx, H in enumerate(Hs):
        H = np.nan_to_num(H)
        H = np.clip(H, 0, np.max(H))
        H_dB = 20 * np.log10(H + 1e-8)
        if zero_mean:
            H_dB -= np.mean(H_dB)
        if "Target" in legend[idx]:
            ax1.plot(freqs, H_dB, linestyle="--", color="k")
        else:
            ax1.plot(freqs, H_dB)

    plt.legend(legend)

    ax1.set_xscale("log")
    ax1.set_ylim([-80, 0])
    ax1.set_xlim([100, 11000])
    plt.title(title)
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency (Hz)")
    plt.grid(c="lightgray", which="both")

    if filename is not None:
        plt.savefig(f"{filename}.png", dpi=300)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close("all")

    return image


def smooth_spectrum(H):
    # apply Savgol filter for smoothed target curve
    return scipy.signal.savgol_filter(H, 1025, 2)


def get_average_spectrum(x, n_fft):
    X = torch.stft(x, n_fft, return_complex=True, normalized=True)
    X = X.abs()  # convert to magnitude
    X = X.mean(dim=-1).view(-1)  # average across frames
    return X
