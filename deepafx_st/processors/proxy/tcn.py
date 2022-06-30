#   Copyright 2022 Christian J. Steinmetz

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# TCN implementation adapted from:
# https://github.com/csteinmetz1/micro-tcn/blob/main/microtcn/tcn.py

import torch
from argparse import ArgumentParser

from deepafx_st.utils import center_crop, causal_crop


class FiLM(torch.nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = torch.nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):

        # project conditioning to 2 x num. conv channels
        cond = self.adaptor(cond)

        # split the projection into gain and bias
        g, b = torch.chunk(cond, 2, dim=-1)

        # add virtual channel dim if needed
        if g.ndim == 2:
            g = g.unsqueeze(1)
            b = b.unsqueeze(1)

        # reshape for application
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        x = self.bn(x)  # apply BatchNorm without affine
        x = (x * g) + b  # then apply conditional affine

        return x


class ConditionalTCNBlock(torch.nn.Module):
    def __init__(
        self, in_ch, out_ch, cond_dim, kernel_size=3, dilation=1, causal=False, **kwargs
    ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal

        self.conv1 = torch.nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            bias=True,
        )
        self.film = FiLM(out_ch, cond_dim)
        self.relu = torch.nn.PReLU(out_ch)
        self.res = torch.nn.Conv1d(
            in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False
        )

    def forward(self, x, p):
        x_in = x

        x = self.conv1(x)
        x = self.film(x, p)  # apply FiLM conditioning
        x = self.relu(x)
        x_res = self.res(x_in)

        if self.causal:
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])

        return x


class ConditionalTCN(torch.nn.Module):
    """Temporal convolutional network with conditioning module.
    Args:
        sample_rate (float): Audio sample rate.
        num_control_params (int, optional): Dimensionality of the conditioning signal. Default: 24
        ninputs (int, optional): Number of input channels (mono = 1, stereo 2). Default: 1
        noutputs (int, optional): Number of output channels (mono = 1, stereo 2). Default: 1
        nblocks (int, optional): Number of total TCN blocks. Default: 10
        kernel_size (int, optional: Width of the convolutional kernels. Default: 3
        dialation_growth (int, optional): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 1
        channel_growth (int, optional): Compute the output channels at each black as in_ch * channel_growth. Default: 2
        channel_width (int, optional): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 64
        stack_size (int, optional): Number of blocks that constitute a single stack of blocks. Default: 10
        causal (bool, optional): Causal TCN configuration does not consider future input values. Default: False
    """

    def __init__(
        self,
        sample_rate,
        num_control_params=24,
        ninputs=1,
        noutputs=1,
        nblocks=10,
        kernel_size=15,
        dilation_growth=2,
        channel_growth=1,
        channel_width=64,
        stack_size=10,
        causal=False,
        skip_connections=False,
        **kwargs,
    ):
        super().__init__()
        self.num_control_params = num_control_params
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.nblocks = nblocks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.channel_growth = channel_growth
        self.channel_width = channel_width
        self.stack_size = stack_size
        self.causal = causal
        self.skip_connections = skip_connections
        self.sample_rate = sample_rate

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = out_ch if n > 0 else ninputs

            if self.channel_growth > 1:
                out_ch = in_ch * self.channel_growth
            else:
                out_ch = self.channel_width

            dilation = self.dilation_growth ** (n % self.stack_size)

            self.blocks.append(
                ConditionalTCNBlock(
                    in_ch,
                    out_ch,
                    self.num_control_params,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    padding="same" if self.causal else "valid",
                    causal=self.causal,
                )
            )

        self.output = torch.nn.Conv1d(out_ch, noutputs, kernel_size=1)
        self.receptive_field = self.compute_receptive_field()
        # print(
        #     f"TCN receptive field: {self.receptive_field} samples",
        #     f" or {(self.receptive_field/self.sample_rate)*1e3:0.3f} ms",
        # )

    def forward(self, x, p, **kwargs):

        # causally pad input signal
        x = torch.nn.functional.pad(x, (self.receptive_field - 1, 0))

        # iterate over blocks passing conditioning
        for idx, block in enumerate(self.blocks):
            x = block(x, p)
            if self.skip_connections:
                if idx == 0:
                    skips = x
                else:
                    skips = center_crop(skips, x[-1]) + x
            else:
                skips = 0

        # final 1x1 convolution to collapse channels
        out = self.output(x + skips)

        return out

    def compute_receptive_field(self):
        """Compute the receptive field in samples."""
        rf = self.kernel_size
        for n in range(1, self.nblocks):
            dilation = self.dilation_growth ** (n % self.stack_size)
            rf = rf + ((self.kernel_size - 1) * dilation)
        return rf
