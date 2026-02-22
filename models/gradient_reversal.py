"""
Gradient Reversal Layer (GRL) for adversarial vocoder training (Section II-A.2).

Reverses gradients with scaling factor lambda_adv = 0.3 during backpropagation,
encouraging the encoder to learn vocoder-invariant representations.

Reference: Ganin & Lempitsky, ICML 2015.
"""

import torch
from torch.autograd import Function


class _GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    """
    Gradient Reversal Layer.

    Forward: identity.
    Backward: reverses gradients scaled by lambda_adv.

    Args:
        lambda_val (float): Gradient reversal scaling factor (default: 0.3).
    """

    def __init__(self, lambda_val: float = 0.3):
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return _GradientReversal.apply(x, self.lambda_val)
