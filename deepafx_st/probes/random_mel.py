import math
import torch
import librosa

# based on https://github.com/neuralaudio/hear-baseline/blob/main/hearbaseline/naive.py


class RandomMelProjection(torch.nn.Module):
    def __init__(
        self,
        sample_rate,
        embed_dim=4096,
        n_mels=128,
        n_fft=4096,
        hop_size=1024,
        seed=0,
        epsilon=1e-4,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.embed_dim = embed_dim
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.seed = seed
        self.epsilon = epsilon

        # Set random seed
        torch.random.manual_seed(self.seed)

        # Create a Hann window buffer to apply to frames prior to FFT.
        self.register_buffer("window", torch.hann_window(self.n_fft))

        # Create a mel filter buffer.
        mel_scale = torch.tensor(
            librosa.filters.mel(
                self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
            )
        )
        self.register_buffer("mel_scale", mel_scale)

        # Projection matrices.
        normalization = math.sqrt(self.n_mels)
        self.projection = torch.nn.Parameter(
            torch.rand(self.n_mels, self.embed_dim) / normalization,
            requires_grad=False,
        )

    def forward(self, x):
        bs, chs, samp = x.size()

        x = torch.stft(
            x.view(bs, -1),
            self.n_fft,
            self.hop_size,
            window=self.window,
            return_complex=True,
        )
        x = x.unsqueeze(1).permute(0, 1, 3, 2)

        # Apply the mel-scale filter to the power spectrum.
        x = torch.matmul(x.abs(), self.mel_scale.transpose(0, 1))

        # power scale
        x = torch.pow(x + self.epsilon, 0.3)

        # apply random projection
        e = x.matmul(self.projection)

        # take mean across temporal dim
        e = e.mean(dim=2).view(bs, -1)

        return e

    def compute_frame_embedding(self, x):
        # Compute the real-valued Fourier transform on windowed input signal.
        x = torch.fft.rfft(x * self.window)

        # Convert to a power spectrum.
        x = torch.abs(x) ** 2.0

        # Apply the mel-scale filter to the power spectrum.
        x = torch.matmul(x, self.mel_scale.transpose(0, 1))

        # Convert to a log mel spectrum.
        x = torch.log(x + self.epsilon)

        # Apply projection to get a 4096 dimension embedding
        embedding = x.matmul(self.projection)

        return embedding
