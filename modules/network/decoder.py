"""
Module containing the Decoder class.
"""

from typing import Any

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Decoder class that reconstructs the input from the encoded features.
    """

    def __init__(self, out_shape: tuple[int, int, int], in_channels: int) -> None:
        """
        Initialize the Decoder.
        Args:
            out_shape: Desired output shape of the decoder (c, h, w).
            in_channels: Number of input channels (features).
        """

        super().__init__()

        # Default values for the layers
        default_layers: tuple[dict[str, int], ...] = ({'in_channels': in_channels,      'out_channels': in_channels * 2,    'kernel_size': 3,   'stride': 2,    'padding': 1},
                                                      {'in_channels': in_channels * 2,  'out_channels': in_channels * 4,    'kernel_size': 9,   'stride': 3,    'padding': 1},
                                                      {'in_channels': in_channels * 4,  'out_channels': in_channels * 4,    'kernel_size': 7,   'stride': 5,    'padding': 1},
                                                      {'in_channels': in_channels * 4,  'out_channels': in_channels * 2,    'kernel_size': 9,   'stride': 2,    'padding': 0},
                                                      {'in_channels': in_channels * 2,  'out_channels': in_channels,        'kernel_size': 6,   'stride': 1,    'padding': 0},
                                                      {'in_channels': in_channels,      'out_channels': out_shape[0],       'kernel_size': 11,  'stride': 1,    'padding': 0}
                                                      )
        
        layers: list[nn.Module] = []
        out_hw: tuple[int, int] = out_shape[1:]
        for i, default_params in enumerate(reversed(default_layers)):

            # Get the correct parameters
            params: dict[str, Any]
            params, out_hw = self.adjust_parameters(default_params, out_hw)

            # Create the deconvolutional block
            if i == 0:
                # The last layer uses Tanh
                deconv_block = nn.Sequential(nn.ConvTranspose2d(**params), nn.Tanh())
            else:
                deconv_block = nn.Sequential(nn.ConvTranspose2d(**params), nn.BatchNorm2d(params['out_channels']), nn.ReLU(inplace = True))
            
            layers.insert(0, deconv_block)
        self.decoder: nn.Sequential = nn.Sequential(*layers)
        self.input_shape: tuple[int, int] = out_hw
            
    @staticmethod
    def adjust_parameters(default_params: dict[str, Any], out_shape: tuple[int, int]) -> tuple[dict[str, Any], tuple[int, int]]:
        """
        Fix the parameters of a ConvTranspose2d layer to ensure the output size is correct.

        Args:
            default_params: Default parameters for the ConvTranspose2d layer. Must include 'kernel_size', 'stride', and optionally 'padding' and 'output_padding'.
            out_shape: Desired output shape (height, width) of the layer.

        Returns:
            A tuple containing the updated parameters and the calculated input size.
        """

        def to_tuple(value: int|tuple[int, int]) -> tuple[int, int]:
            """
            Expand an integer to a tuple if needed.
            """
            if isinstance(value, int):
                return (value, value)
            return value

        # Extract the parameters
        kernel_size: tuple[int, int] = to_tuple(default_params['kernel_size'])
        stride: tuple[int, int] = to_tuple(default_params['stride'])
        padding: tuple[int, int] = to_tuple(default_params.get('padding', 0))
        output_padding: tuple[int, int] = to_tuple(default_params.get('output_padding', 0))

        # Loop through the dimensions
        new_padding: list[int] = list(padding)
        new_output_padding: list[int] = list(output_padding)
        input_shape: list[int] = [0, 0] # placeholder
        for dim in range(2):
            out_dim: int = out_shape[dim]
            kernel_dim: int = kernel_size[dim]
            stride_dim: int = stride[dim]
            padding_dim: int = padding[dim]
            output_padding_dim: int = output_padding[dim]

            # If the default parameters are not valid, update padding and output_padding
            if (out_dim - kernel_dim + 2 * padding_dim - output_padding_dim) % stride_dim != 0:
                size_mismatch: int = (kernel_dim + stride_dim - out_dim) % stride_dim
                output_padding_dim = size_mismatch % 2
                padding_dim = (size_mismatch + output_padding_dim) // 2

            input_dim: int = 1 + ((out_dim - kernel_dim + 2 * padding_dim - output_padding_dim) // stride_dim)

            # If the desired output is too small, increase the padding
            if input_dim <= 0:
                padding_dim += (stride_dim * (2 - input_dim)) // 2
                input_dim = 1
            
            # Update the values
            new_padding[dim] = padding_dim
            new_output_padding[dim] = output_padding_dim
            input_shape[dim] = input_dim

        # Update the default parameters with the calculated ones
        params: dict[str, Any] = default_params.copy()
        params['padding'] = tuple(new_padding)
        params['output_padding'] = tuple(new_output_padding)

        return params, tuple(input_shape)   # type: ignore


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


    def get_input_shape(self) -> tuple[int, int]:
        """
        Get the input shape of the decoder.
        """
        return self.input_shape
