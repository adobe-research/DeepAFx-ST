import torch
import numpy as np
import torch.multiprocessing as mp

from deepafx_st.processors.dsp.peq import ParametricEQ
from deepafx_st.processors.dsp.compressor import Compressor
from deepafx_st.processors.spsa.spsa_func import SPSAFunction
from deepafx_st.utils import rademacher


def dsp_func(x, p, dsp, sample_rate=24000):

    (peq, comp), meta = dsp

    p_peq = p[:meta]
    p_comp = p[meta:]

    y = peq(x, p_peq, sample_rate)
    y = comp(y, p_comp, sample_rate)

    return y


class SPSAChannel(torch.nn.Module):
    """

    Args:
        sample_rate (float): Sample rate of the plugin instance
        parallel (bool, optional): Use parallel workers for DSP.

    By default, this utilizes parallelized instances of the plugin channel,
    where the number of workers is equal to the batch size.
    """

    def __init__(
        self,
        sample_rate: int,
        parallel: bool = False,
        batch_size: int = 8,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.parallel = parallel

        if self.parallel:
            self.apply_func = SPSAFunction.apply

            procs = {}
            for b in range(self.batch_size):

                peq = ParametricEQ(sample_rate)
                comp = Compressor(sample_rate)
                dsp = ((peq, comp), peq.num_control_params)

                parent_conn, child_conn = mp.Pipe()
                p = mp.Process(target=SPSAChannel.worker_pipe, args=(child_conn, dsp))
                p.start()
                procs[b] = [p, parent_conn, child_conn]
                #print(b, p)

                # Update stuff for external public members TODO: fix
                self.ports = [peq.ports, comp.ports]
                self.num_control_params = (
                    comp.num_control_params + peq.num_control_params
                )

            self.procs = procs
            #print(self.procs)

        else:
            self.peq = ParametricEQ(sample_rate)
            self.comp = Compressor(sample_rate)
            self.apply_func = SPSAFunction.apply
            self.ports = [self.peq.ports, self.comp.ports]
            self.num_control_params = (
                self.comp.num_control_params + self.peq.num_control_params
            )
            self.dsp = ((self.peq, self.comp), self.peq.num_control_params)

        # add one param for wet/dry mix
        # self.num_control_params += 1

    def __del__(self):
        if hasattr(self, "procs"):
            for proc_idx, proc in self.procs.items():
                #print(f"Closing {proc_idx}...")
                proc[0].terminate()

    def forward(self, x, p, epsilon=0.001, sample_rate=24000, **kwargs):
        """
        Args:
            x (Tensor): Input signal with shape: [batch x channels x samples]
            p (Tensor): Audio effect control parameters with shape: [batch x parameters]
            epsilon (float, optional): Twiddle parameter range for SPSA gradient estimation.

        Returns:
            y (Tensor): Processed audio signal.

        """
        if self.parallel:
            y = self.apply_func(x, p, None, epsilon, self, sample_rate)

        else:
            # this will process on CPU in NumPy
            y = self.apply_func(x, p, None, epsilon, self, sample_rate)

        return y.type_as(x)

    @staticmethod
    def static_backward(dsp, value):

        (
            batch_index,
            x,
            params,
            needs_input_grad,
            needs_param_grad,
            grad_output,
            epsilon,
        ) = value

        grads_input = None
        grads_params = None
        ps = params.shape[-1]
        factors = [1.0]

        # estimate gradient w.r.t input
        if needs_input_grad:
            delta_k = rademacher(x.shape).numpy()
            J_plus = dsp_func(x + epsilon * delta_k, params, dsp)
            J_minus = dsp_func(x - epsilon * delta_k, params, dsp)
            grads_input = (J_plus - J_minus) / (2.0 * epsilon)

        # estimate gradient w.r.t params
        grads_params_runs = []
        if needs_param_grad:
            for factor in factors:
                params_sublist = []
                delta_k = rademacher(params.shape).numpy()

                # compute output in two random directions of the parameter space
                params_plus = np.clip(params + (factor * epsilon * delta_k), 0, 1)
                J_plus = dsp_func(x, params_plus, dsp)

                params_minus = np.clip(params - (factor * epsilon * delta_k), 0, 1)
                J_minus = dsp_func(x, params_minus, dsp)
                grad_param = J_plus - J_minus

                # compute gradient for each parameter as a function of epsilon and random direction
                for sub_p_idx in range(ps):
                    grad_p = grad_param / (2 * epsilon * delta_k[sub_p_idx])
                    params_sublist.append(np.sum(grad_output * grad_p))

                grads_params = np.array(params_sublist)
                grads_params_runs.append(grads_params)

            # average gradients
            grads_params = np.mean(grads_params_runs, axis=0)

        return grads_input, grads_params

    @staticmethod
    def static_forward(dsp, value):
        batch_index, x, p, sample_rate = value
        y = dsp_func(x, p, dsp, sample_rate)
        return y

    @staticmethod
    def worker_pipe(child_conn, dsp):

        while True:
            msg, value = child_conn.recv()
            if msg == "forward":
                child_conn.send(SPSAChannel.static_forward(dsp, value))
            elif msg == "backward":
                child_conn.send(SPSAChannel.static_backward(dsp, value))
            elif msg == "shutdown":
                break
