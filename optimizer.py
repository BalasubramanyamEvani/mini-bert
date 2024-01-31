from typing import Callable, Iterable, Tuple
import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                # State should be stored in this dictionary
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["mt"] = torch.zeros_like(p.data)
                    state["vt"] = torch.zeros_like(p.data)

                mt, vt = state["mt"], state["vt"]
                beta1, beta2 = group["betas"]
                state["t"] += 1

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]

                # Update first and second moments of the gradients
                mt = torch.add(torch.mul(mt, beta1), grad, alpha=1.0 - beta1)
                vt = torch.add(torch.mul(vt, beta2), grad**2, alpha=1.0 - beta2)

                state["mt"] = mt
                state["vt"] = vt

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                if group["correct_bias"]:
                    alpha = alpha * (
                        math.sqrt(1.0 - beta2 ** state["t"])
                        / (1.0 - beta1 ** state["t"])
                    )

                # Update parameters
                p.data = torch.addcdiv(
                    p.data,
                    mt,
                    torch.add(torch.sqrt(vt), group["eps"]),
                    value=-alpha,
                )

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                if group["weight_decay"] > 0.0:
                    p.data = torch.add(
                        p.data, p.data, alpha=-group["lr"] * group["weight_decay"]
                    )
        return loss
