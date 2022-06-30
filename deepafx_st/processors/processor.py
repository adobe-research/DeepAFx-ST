import torch
import multiprocessing
from abc import ABC, abstractmethod
import deepafx_st.utils as utils
import numpy as np


class Processor(torch.nn.Module, ABC):
    """Processor base class."""

    def __init__(
        self,
    ):
        super().__init__()

    def denormalize_params(self, p):
        """This method takes a tensor of parameters scaled from 0-1 and
        restores them back to the original parameter range."""

        # check if the number of parameters is correct
        params = p  # torch.split(p, 1, -1)
        if len(params) != self.num_control_params:
            raise RuntimeError(
                f"Invalid number of parameters. ",
                f"Expected {self.num_control_params} but found {len(params)} {params.shape}.",
            )

        # iterate over the parameters and expand from 0-1 to full range
        denorm_params = []
        for param, port in zip(params, self.ports):
            # check if parameter exceeds range
            if param > 1.0 or param < 0.0:
                raise RuntimeError(
                    f"""Parameter '{port["name"]}' exceeds range: {param}"""
                )

            # denormalize and store result
            denorm_params.append(utils.denormalize(param, port["max"], port["min"]))

        return denorm_params

    def normalize_params(self, *params):
        """This method creates a vector of parameters normalized from 0-1."""

        # check if the number of parameters is correct
        if len(params) != self.num_control_params:
            raise RuntimeError(
                f"Invalid number of parameters. ",
                f"Expected {self.num_control_params} but found {len(params)}.",
            )

        norm_params = []
        for param, port in zip(params, self.ports):
            norm_params.append(utils.normalize(param, port["max"], port["min"]))

        p = torch.tensor(norm_params).view(1, -1)

        return p

    # def run_series(self, inputs, params):
    #    """Run the process function in a loop given a list of inputs and parameters"""
    #    p_b_denorm = [p for p in self.denormalize_params(params)]
    #    y = self.process_fn(inputs, self.sample_rate, *p_b_denorm)
    #    return y

    def run_series(self, inputs, params, sample_rate=24000):
        """Run the process function in a loop given a list of inputs and parameters"""
        if params.ndim == 1:
            params = np.reshape(params, (1, -1))
            inputs = np.reshape(inputs, (1, -1))
        bs = inputs.shape[0]
        ys = []
        params = np.clip(params, 0, 1)
        for bidx in range(bs):
            p_b_denorm = [p for p in self.denormalize_params(params[bidx, :])]
            y = self.process_fn(
                inputs[bidx, ...].reshape(-1),
                sample_rate,
                *p_b_denorm,
            )
            ys.append(y)
        y = np.stack(ys, axis=0)
        return y

    @abstractmethod
    def forward(self, x, p):
        pass
