"""
Module containing the MDN class, which implements a Mixture Density Network.
"""

import torch
import torch.nn as nn


class MDN(nn.Module):
    """
    Mixture Density Network (MDN) that predicts a mixture of Gaussian distributions.
    """

    def __init__(self, input_dim: int, n_components: int) -> None:
        """
        Initialize the Mixture Density Network.

        Args:
            input_dim: Dimension of the input features.
            n_components: Number of Gaussian components in the mixture.
        """

        super().__init__()

        # Parameters
        self.n_components: int = n_components
        
        # Layers
        self.pi_layer: nn.Sequential = nn.Sequential(nn.Linear(input_dim, n_components, bias = False),
                                                     nn.Softmax(dim = -1)
                                                     )
        self.mu_layer: nn.Linear = nn.Linear(input_dim, input_dim * n_components, bias = False)
        self.sigma_layer: nn.Sequential = nn.Sequential(nn.Linear(input_dim, input_dim * n_components, bias = False),
                                                        nn.Softplus()
                                                        )
    

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Compute the parameters of the Mixture Density Network
        pi: torch.Tensor = self.pi_layer(x)
        mu: torch.Tensor = self.mu_layer(x)
        sigma: torch.Tensor = self.sigma_layer(x)

        # Reshape
        mu = mu.view(*x.size(), self.n_components)
        sigma = sigma.view(*x.size(), self.n_components)

        return pi, mu, sigma
