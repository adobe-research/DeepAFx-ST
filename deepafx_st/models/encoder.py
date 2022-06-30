import torch

from deepafx_st.models.mobilenetv2 import MobileNetV2
from deepafx_st.models.efficient_net import EfficientNet


class SpectralEncoder(torch.nn.Module):
    def __init__(
        self,
        num_params,
        sample_rate,
        encoder_model="mobilenet_v2",
        embed_dim=1028,
        width_mult=1,
        min_level_db=-80,
    ):
        """Encoder operating on spectrograms.

        Args:
            num_params (int): Number of processor parameters to generate.
            sample_rate (float): Audio sample rate for computing melspectrogram.
            encoder_model (str, optional): Encoder model architecture. Default: "mobilenet_v2"
            embed_dim (int, optional): Dimentionality of the encoder representations.
            width_mult (int, optional): Encoder size. Default: 1
            min_level_db (float, optional): Minimal dB value for the spectrogram. Default: -80
        """
        super().__init__()
        self.num_params = num_params
        self.sample_rate = sample_rate
        self.encoder_model = encoder_model
        self.embed_dim = embed_dim
        self.width_mult = width_mult
        self.min_level_db = min_level_db

        # load model from torch.hub
        if encoder_model == "mobilenet_v2":
            self.encoder = MobileNetV2(embed_dim=embed_dim, width_mult=width_mult)
        elif encoder_model == "efficient_net":
            self.encoder = EfficientNet.from_name(
                "efficientnet-b2",
                in_channels=1,
                image_size=(128, 65),
                include_top=False,
            )
            self.embedding_projection = torch.nn.Conv2d(
                in_channels=1408,
                out_channels=embed_dim,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=True,
            )

        else:
            raise ValueError(f"Invalid encoder_model: {encoder_model}.")

        self.window = torch.nn.Parameter(torch.hann_window(4096))

    def forward(self, x):
        """
        Args:
            x (Tensor): Input waveform of shape [batch x channels x samples]

        Returns:
            e (Tensor): Latent embedding produced by Encoder. [batch x embed_dim]
        """
        bs, chs, samp = x.size()

        # compute spectrogram of waveform
        X = torch.stft(
            x.view(bs, -1),
            4096,
            2048,
            window=self.window,
            return_complex=True,
        )
        X_db = torch.pow(X.abs() + 1e-8, 0.3)
        X_db_norm = X_db

        # standardize (0, 1) 0.322970 0.278452
        X_db_norm -= 0.322970
        X_db_norm /= 0.278452
        X_db_norm = X_db_norm.unsqueeze(1).permute(0, 1, 3, 2)

        if self.encoder_model == "mobilenet_v2":
            # repeat channels by 3 to fit vision model
            X_db_norm = X_db_norm.repeat(1, 3, 1, 1)

            # pass melspectrogram through encoder
            e = self.encoder(X_db_norm)

            # apply avg pooling across time for encoder embeddings
            e = torch.nn.functional.adaptive_avg_pool2d(e, 1).reshape(e.shape[0], -1)

            # normalize by L2 norm
            norm = torch.norm(e, p=2, dim=-1, keepdim=True)
            e_norm = e / norm

        elif self.encoder_model == "efficient_net":

            # Efficient Net internal downsamples by 32 on time and freq axis, then average pools the rest
            e = self.encoder(X_db_norm)

            # Adding 1x1 conv to project down or up to the requested embedding size
            e = self.embedding_projection(e)
            e = torch.squeeze(e, dim=3)
            e = torch.squeeze(e, dim=2)

            # normalize by L2 norm
            norm = torch.norm(e, p=2, dim=-1, keepdim=True)
            e_norm = e / norm

        return e_norm
