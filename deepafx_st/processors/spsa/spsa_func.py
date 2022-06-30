import torch


def spsa_func(input, params, process, i, sample_rate=24000):
    return process(input.cpu(), params.cpu(), i, sample_rate).type_as(input)


class SPSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input,
        params,
        process,
        epsilon,
        thread_context,
        sample_rate=24000,
    ):
        """Apply processor to a batch of tensors using given parameters.

        Args:
            input (Tensor): Audio with shape: batch x 2 x samples
            params (Tensor): Processor parameters with shape: batch x params
            process (function): Function that will apply processing.
            epsilon (float): Perturbation strength for SPSA computation.

        Returns:
            output (Tensor): Processed audio with same shape as input.
        """
        ctx.save_for_backward(input, params)
        ctx.epsilon = epsilon
        ctx.process = process
        ctx.thread_context = thread_context

        if thread_context.parallel:

            for i in range(input.shape[0]):
                msg = (
                    "forward",
                    (
                        i,
                        input[i].view(-1).detach().cpu().numpy(),
                        params[i].view(-1).detach().cpu().numpy(),
                        sample_rate,
                    ),
                )
                thread_context.procs[i][1].send(msg)

            z = torch.empty_like(input)
            for i in range(input.shape[0]):
                z[i] = torch.from_numpy(thread_context.procs[i][1].recv())
        else:
            z = torch.empty_like(input)
            for i in range(input.shape[0]):
                value = (
                    i,
                    input[i].view(-1).detach().cpu().numpy(),
                    params[i].view(-1).detach().cpu().numpy(),
                    sample_rate,
                )
                z[i] = torch.from_numpy(
                    thread_context.static_forward(thread_context.dsp, value)
                )

        return z

    @staticmethod
    def backward(ctx, grad_output):
        """Estimate gradients using SPSA."""

        input, params = ctx.saved_tensors
        epsilon = ctx.epsilon
        needs_input_grad = ctx.needs_input_grad[0]
        needs_param_grad = ctx.needs_input_grad[1]
        thread_context = ctx.thread_context

        grads_input = None
        grads_params = None

        # Receive grads
        if needs_input_grad:
            grads_input = torch.empty_like(input)
        if needs_param_grad:
            grads_params = torch.empty_like(params)

        if thread_context.parallel:

            for i in range(input.shape[0]):
                msg = (
                    "backward",
                    (
                        i,
                        input[i].view(-1).detach().cpu().numpy(),
                        params[i].view(-1).detach().cpu().numpy(),
                        needs_input_grad,
                        needs_param_grad,
                        grad_output[i].view(-1).detach().cpu().numpy(),
                        epsilon,
                    ),
                )
                thread_context.procs[i][1].send(msg)

            # Wait for output
            for i in range(input.shape[0]):
                temp1, temp2 = thread_context.procs[i][1].recv()

                if temp1 is not None:
                    grads_input[i] = torch.from_numpy(temp1)

                if temp2 is not None:
                    grads_params[i] = torch.from_numpy(temp2)

            return grads_input, grads_params, None, None, None, None
        else:
            for i in range(input.shape[0]):
                value = (
                    i,
                    input[i].view(-1).detach().cpu().numpy(),
                    params[i].view(-1).detach().cpu().numpy(),
                    needs_input_grad,
                    needs_param_grad,
                    grad_output[i].view(-1).detach().cpu().numpy(),
                    epsilon,
                )
                temp1, temp2 = thread_context.static_backward(thread_context.dsp, value)
                if temp1 is not None:
                    grads_input[i] = torch.from_numpy(temp1)

                if temp2 is not None:
                    grads_params[i] = torch.from_numpy(temp2)
            return grads_input, grads_params, None, None, None, None
