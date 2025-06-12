"""
Module containing the Capsule class, which implements a capsule network layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Capsule(nn.Module):
    """
    Capsule layer that applies dynamic routing to the input tensor.
    """

    def __init__(self, n_caps: int, caps_dim: int, output_dim: int, iterations: int) -> None:
        """
        Initialize the Capsule layer.
        Args:
            n_caps: Number of capsules in the layer.
            caps_dim: Dimension of each capsule.
            output_dim: Output dimension of the capsules.
            iterations: Number of routing iterations to perform.
        """

        super().__init__()

        # Parameters
        self.iterations: int = iterations
        self.n_caps: int = n_caps

        # Capsule weights
        self.caps_weights: torch.Tensor = nn.Parameter(0.01 * torch.randn(n_caps, caps_dim, output_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Project the input to capsule space
        x = torch.einsum('n c o, p b o -> b n o', self.caps_weights, x)
        x_detached: torch.Tensor = x.detach()

        # Routing
        logits: torch.Tensor = torch.zeros(x.size(0), 1, self.n_caps, device = x.device)
        coeffs: torch.Tensor = F.softmax(logits, dim = -1)
        for _ in range(self.iterations - 1):
            routed_caps: torch.Tensor = self.squash(coeffs @ x_detached)
            logits += routed_caps @ x_detached.mT
            coeffs: torch.Tensor = F.softmax(logits, dim = -1)

        return self.squash(coeffs @ x)

    @staticmethod
    def squash(x: torch.Tensor) -> torch.Tensor:
        """
        Squashing function to normalize the capsule outputs.
        """

        norm: torch.Tensor = torch.norm(x, dim = -1, keepdim = True)
        scale: torch.Tensor = norm / (1 + norm ** 2)
        return scale * x
