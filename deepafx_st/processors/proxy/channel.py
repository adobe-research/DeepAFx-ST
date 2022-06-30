import torch
from deepafx_st.processors.proxy.proxy_system import ProxySystem
from deepafx_st.utils import DSPMode


class ProxyChannel(torch.nn.Module):
    def __init__(
        self,
        proxy_system_ckpts: list,
        freeze_proxies: bool = True,
        dsp_mode: DSPMode = DSPMode.NONE,
        num_tcns: int = 2,
        tcn_nblocks: int = 4,
        tcn_dilation_growth: int = 8,
        tcn_channel_width: int = 64,
        tcn_kernel_size: int = 13,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.freeze_proxies = freeze_proxies
        self.dsp_mode = dsp_mode
        self.num_tcns = num_tcns

        # load the proxies
        self.proxies = torch.nn.ModuleList()
        self.num_control_params = 0
        self.ports = []
        for proxy_system_ckpt in proxy_system_ckpts:
            proxy = ProxySystem.load_from_checkpoint(proxy_system_ckpt)
            # freeze model parameters
            if freeze_proxies:
                for param in proxy.parameters():
                    param.requires_grad = False
            self.proxies.append(proxy)
            if proxy.hparams.processor == "channel":
                self.ports = proxy.processor.ports
            else:
                self.ports.append(proxy.processor.ports)
            self.num_control_params += proxy.processor.num_control_params

        if len(proxy_system_ckpts) == 0:
            if self.num_tcns == 2:
                peq_proxy = ProxySystem(
                    processor="peq",
                    output_gain=False,
                    nblocks=tcn_nblocks,
                    dilation_growth=tcn_dilation_growth,
                    kernel_size=tcn_kernel_size,
                    channel_width=tcn_channel_width,
                    sample_rate=sample_rate,
                )
                self.proxies.append(peq_proxy)
                self.ports.append(peq_proxy.processor.ports)
                self.num_control_params += peq_proxy.processor.num_control_params
                comp_proxy = ProxySystem(
                    processor="comp",
                    output_gain=True,
                    nblocks=tcn_nblocks,
                    dilation_growth=tcn_dilation_growth,
                    kernel_size=tcn_kernel_size,
                    channel_width=tcn_channel_width,
                    sample_rate=sample_rate,
                )
                self.proxies.append(comp_proxy)
                self.ports.append(comp_proxy.processor.ports)
                self.num_control_params += comp_proxy.processor.num_control_params
            elif self.num_tcns == 1:
                channel_proxy = ProxySystem(
                    processor="channel",
                    output_gain=True,
                    nblocks=tcn_nblocks,
                    dilation_growth=tcn_dilation_growth,
                    kernel_size=tcn_kernel_size,
                    channel_width=tcn_channel_width,
                    sample_rate=sample_rate,
                )
                self.proxies.append(channel_proxy)
                for port_list in channel_proxy.processor.ports:
                    self.ports.append(port_list)
                self.num_control_params += channel_proxy.processor.num_control_params
            else:
                raise ValueError(f"num_tcns must be <= 2. Asked for {self.num_tcns}.")

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        dsp_mode: DSPMode = DSPMode.NONE,
        sample_rate: int = 24000,
        **kwargs,
    ):
        # loop over the proxies and pass parameters
        stop_idx = 0
        for proxy in self.proxies:
            start_idx = stop_idx
            stop_idx += proxy.processor.num_control_params
            p_subset = p[:, start_idx:stop_idx]
            if dsp_mode.name == DSPMode.NONE.name:
                x = proxy(
                    x,
                    p_subset,
                    use_dsp=False,
                )
            elif dsp_mode.name == DSPMode.INFER.name:
                x = proxy(
                    x,
                    p_subset,
                    use_dsp=True,
                    sample_rate=sample_rate,
                )
            elif dsp_mode.name == DSPMode.TRAIN_INFER.name:
                # Mimic gumbel softmax implementation to replace grads similar to
                # https://gist.github.com/yzh119/fd2146d2aeb329d067568a493b20172f
                x_hard = proxy(
                    x,
                    p_subset,
                    use_dsp=True,
                    sample_rate=sample_rate,
                )
                x = proxy(
                    x,
                    p_subset,
                    use_dsp=False,
                    sample_rate=sample_rate,
                )
                x = (x_hard - x).detach() + x
            else:
                assert 0, "invalid dsp model for proxy"

        return x
