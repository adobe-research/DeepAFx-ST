# MIT License

# Copyright (c) 2021 Pranay Manocha

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# code adapated from https://github.com/pranaymanocha/PerceptualAudio

import cdpam
import torch


class CDPAMEncoder(torch.nn.Module):
    def __init__(self, cdpam_ckpt: str):
        super().__init__()

        # pre-trained model parameterss
        encoder_layers = 16
        encoder_filters = 64
        input_size = 512
        proj_ndim = [512, 256]
        ndim = [16, 6]
        classif_BN = 0
        classif_act = "no"
        proj_dp = 0.1
        proj_BN = 1
        classif_dp = 0.05

        model = cdpam.models.FINnet(
            encoder_layers=encoder_layers,
            encoder_filters=encoder_filters,
            ndim=ndim,
            classif_dp=classif_dp,
            classif_BN=classif_BN,
            classif_act=classif_act,
            input_size=input_size,
        )

        state = torch.load(cdpam_ckpt, map_location="cpu")["state"]
        model.load_state_dict(state)
        model.eval()

        self.model = model
        self.embed_dim = 512

    def forward(self, x):

        with torch.no_grad():
            _, a1, c1 = self.model.base_encoder.forward(x)
            a1 = torch.nn.functional.normalize(a1, dim=1)

        return a1
