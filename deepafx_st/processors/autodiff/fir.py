import torch


class FIRFilter(torch.nn.Module):
    def __init__(self, num_control_params=63):
        super().__init__()
        self.num_control_params = num_control_params
        self.adaptor = torch.nn.Linear(num_control_params, num_control_params)
        #self.batched_lfilter = torch.vmap(self.lfilter)

    def forward(self, x, b, **kwargs):
        """Forward pass by appling FIR filter to each batch element.

        Args:
            x (tensor): Input signals with shape (batch x 1 x samples)
            b (tensor): Matrix of FIR filter coefficients with shape (batch x ntaps)

        """
        bs, ch, s = x.size()
        b = self.adaptor(b)

        # pad input
        x = torch.nn.functional.pad(x, (b.shape[-1] // 2, b.shape[-1] // 2))

        # add extra dim for virutal batch dim
        x = x.view(bs, 1, ch, -1)
        b = b.view(bs, 1, 1, -1)

        # exlcuding vmap for now
        y = self.batched_lfilter(x, b).view(bs, ch, s)

        return y

    @staticmethod
    def lfilter(x, b):
        return torch.nn.functional.conv1d(x, b)


class FrequencyDomainFIRFilter(torch.nn.Module):
    def __init__(self, num_control_params=31):
        super().__init__()
        self.num_control_params = num_control_params
        self.adaptor = torch.nn.Linear(num_control_params, num_control_params)

    def forward(self, x, b, **kwargs):
        """Forward pass by appling FIR filter to each batch element.

        Args:
            x (tensor): Input signals with shape (batch x 1 x samples)
            b (tensor): Matrix of FIR filter coefficients with shape (batch x ntaps)
        """
        bs, c, s = x.size()

        b = self.adaptor(b)

        # transform input to freq. domain
        X = torch.fft.rfft(x.view(bs, -1))

        # frequency response of filter
        H = torch.fft.rfft(b.view(bs, -1))

        # apply filter as multiplication in freq. domain
        Y = X * H

        # transform back to time domain
        y = torch.fft.ifft(Y).view(bs, 1, -1)

        return y
