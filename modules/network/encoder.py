"""
Module containing the ViTEncoder class, which implements a Vision Transformer (ViT) encoder.
"""

import einops
import torch
import torch.nn as nn


class DyTanh(nn.Module):
    """
    DyTanh normalization layer.
    """

    def __init__(self, dim: int, alpha: float = 0.5) -> None:
        """
        Initialize the DyTanh layer.

        Args:
            dim: Dimension of the input tensor.
            alpha: Starting scaling factor to apply before the tanh.
        """

        super().__init__()
        
        self.alpha: torch.Tensor = nn.Parameter(torch.full((dim,), alpha))
        self.tanh: nn.Tanh = nn.Tanh()
        self.weights: torch.Tensor = nn.Parameter(torch.ones(dim))
        self.bias: torch.Tensor = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.tanh(x * self.alpha)
        return self.weights * x + self.bias
    

class ViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder.
    """

    def __init__(self,
                 patch_shape: tuple[int, int],
                 n_patches: int,
                 n_channels: int,
                 embedding_dim: int,
                 heads: int,
                 ff_dim: int,
                 depth: int,
                 use_dytanh: bool = False
                 ) -> None:
        """
        Initialize the ViT encoder.
        
        Args:
            patch_shape: Shape of the patches to split the input image.
            n_patches: Number of patches in the input image.
            n_channels: Number of channels in the input image.
            embedding_dim: Dimension of the embedding space.
            heads: Number of attention heads in the transformer.
            ff_dim: Dimension of the feedforward network in the transformer.
            depth: Number of layers in the transformer encoder.
            use_dytanh: Whether to use DyTanh normalization instead of LayerNorm.
        """

        super().__init__()

        # Split in patches and embed
        self.embed_image: nn.Conv2d = nn.Conv2d(n_channels,
                                                out_channels = embedding_dim,
                                                kernel_size = patch_shape,
                                                stride = patch_shape
                                                )
        self.pos_embedding: torch.Tensor = nn.Parameter(torch.randn(n_patches, 1, embedding_dim))

        # Transformer encoder
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model = embedding_dim,
                                                                               nhead = heads,
                                                                               dim_feedforward = ff_dim,
                                                                               activation = 'gelu',
                                                                               norm_first = True,
                                                                               )
        if use_dytanh:
            encoder_layer.norm1 = DyTanh(embedding_dim) # type: ignore
            encoder_layer.norm2 = DyTanh(embedding_dim) # type: ignore
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers = depth)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Embedding
        x = self.embed_image(x)
        x = einops.rearrange(x, 'b e h w -> (h w) b e')

        # Transformer
        return self.transformer(x + self.pos_embedding)
