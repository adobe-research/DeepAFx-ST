import torch

class StyleTransferController(torch.nn.Module):
    def __init__(
        self,
        num_control_params,
        edim,
        hidden_dim=256,
        agg_method="mlp",
    ):
        """Plugin parameter controller module to map from input to target style.

        Args:
            num_control_params (int): Number of plugin parameters to predicted.
            edim (int): Size of the encoder representations.
            hidden_dim (int, optional): Hidden size of the 3-layer parameter predictor MLP. Default: 256
            agg_method (str, optional): Input/reference embed aggregation method ["conv" or "linear", "mlp"]. Default: "mlp"
        """
        super().__init__()
        self.num_control_params = num_control_params
        self.edim = edim
        self.hidden_dim = hidden_dim
        self.agg_method = agg_method

        if agg_method == "conv":
            self.agg = torch.nn.Conv1d(
                2,
                1,
                kernel_size=129,
                stride=1,
                padding="same",
                bias=False,
            )
            mlp_in_dim = edim
        elif agg_method == "linear":
            self.agg = torch.nn.Linear(edim * 2, edim)
        elif agg_method == "mlp":
            self.agg = None
            mlp_in_dim = edim * 2
        else:
            raise ValueError(f"Invalid agg_method = {self.agg_method}.")

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in_dim, hidden_dim),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(hidden_dim, num_control_params),
            torch.nn.Sigmoid(),  # normalize between 0 and 1
        )

    def forward(self, e_x, e_y, z=None):
        """Forward pass to generate plugin parameters.

        Args:
            e_x (tensor): Input signal embedding of shape (batch, edim)
            e_y (tensor): Target signal embedding of shape (batch, edim)
        Returns:
            p (tensor): Estimated control parameters of shape (batch, num_control_params)
        """

        # use learnable projection
        if self.agg_method == "conv":
            e_xy = torch.stack((e_x, e_y), dim=1)  # concat on channel dim
            e_xy = self.agg(e_xy)
        elif self.agg_method == "linear":
            e_xy = torch.cat((e_x, e_y), dim=-1)  # concat on embed dim
            e_xy = self.agg(e_xy)
        else:
            e_xy = torch.cat((e_x, e_y), dim=-1)  # concat on embed dim

        # pass through MLP to project to control parametesr
        p = self.mlp(e_xy.squeeze(1))

        return p
