import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import deepafx_st.utils as utils


class LogParametersCallback(pl.callbacks.Callback):
    def __init__(self, num_examples=4):
        super().__init__()
        self.num_examples = 4

    def on_validation_epoch_start(self, trainer, pl_module):
        """At the start of validation init storage for parameters."""
        self.params = []

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        """Called when the validation batch ends.

        Here we log the parameters only from the first batch.

        """
        if outputs is not None and batch_idx == 0:
            examples = np.min([self.num_examples, outputs["x"].shape[0]])
            for n in range(examples):
                self.log_parameters(
                    outputs,
                    n,
                    pl_module.processor.ports,
                    trainer.global_step,
                    trainer.logger,
                    True if batch_idx == 0 else False,
                )

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def log_parameters(self, outputs, batch_idx, ports, global_step, logger, log=True):
        p = outputs["p"][batch_idx, ...]

        table = ""

        # table += f"""## {plugin["name"]}\n"""
        table += "| Index| Name | Value | Units | Min | Max | Default | Raw Value | \n"
        table += "|------|------|------:|:------|----:|----:|--------:| ---------:| \n"

        start_idx = 0
        # set plugin parameters based on provided normalized parameters
        for port_list in ports:
            for pidx, port in enumerate(port_list):
                param_max = port["max"]
                param_min = port["min"]
                param_name = port["name"]
                param_default = port["default"]
                param_units = port["units"]

                param_val = p[start_idx]
                denorm_val = utils.denormalize(param_val, param_max, param_min)

                # add values to table in row
                table += f"| {start_idx + 1} | {param_name} "
                if np.abs(denorm_val) > 10:
                    table += f"| {denorm_val:0.1f} "
                    table += f"| {param_units} "
                    table += f"| {param_min:0.1f} | {param_max:0.1f} "
                    table += f"| {param_default:0.1f} "
                else:
                    table += f"| {denorm_val:0.3f} "
                    table += f"| {param_units} "
                    table += f"| {param_min:0.3f} | {param_max:0.3f} "
                    table += f"| {param_default:0.3f} "

                table += f"| {np.squeeze(param_val):0.2f} | \n"
                start_idx += 1

        table += "\n\n"

        if log:
            logger.experiment.add_text(f"params/{batch_idx+1}", table, global_step)
