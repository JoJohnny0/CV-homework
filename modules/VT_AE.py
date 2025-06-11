"""
Module containing the Vision Transformer Autoencoder (VT-AE) model, available in the VTAE class.
"""

from collections.abc import Callable

import einops
import lightning.pytorch as pl
import torch
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchvision.transforms.functional import gaussian_blur


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


class VTAE(pl.LightningModule):
    """
    Vision Transformer Autoencoder (VT-AE) model for anomaly detection.
    """

    def __init__(self,
                 patch_shape: tuple[int, int],
                 embedding_dim: int,
                 heads: int,
                 depth: int,
                 caps_per_patch: int,
                 caps_dim: int,
                 caps_depth: int,
                 ff_dim: int,
                 coefs: int,
                 noise: float,
                 lr: float = 1e-3,
                 use_dytanh: bool = False
                 ) -> None:
        """
        Initialize the VTAE model. Input must be a 3-channel image of size (512, 512).

        Args:
            patch_shape: Shape of the patches used in the ViT encoder.
            embedding_dim: Dimension of the embedding space.
            heads: Number of attention heads in the transformer encoder.
            depth: Number of transformer encoder layers.
            caps_per_patch: Number of capsules per patch.
            caps_dim: Dimension of the capsules in the capsule layer.
            caps_depth: Depth of the capsule layer.
            ff_dim: Dimension of the feedforward network in the transformer encoder.
            coefs: Number of mixture components in the mixture density network.
            noise: Standard deviation of the Gaussian noise added to the features during training.
            lr: Learning rate for the optimizer.
            use_dytanh: Whether to use dyTanh normalization. If not, use layer normalization.
        """

        super().__init__()
        self.save_hyperparameters()

        self.image_side: int = 512
        self.channels: int = 3

        # ViT encoder
        self.n_patches: int = (self.image_side // patch_shape[0]) * (self.image_side // patch_shape[1])

        self.embed_image: nn.Conv2d = nn.Conv2d(self.channels,
                                                out_channels = embedding_dim,
                                                kernel_size = patch_shape,
                                                stride = patch_shape
                                                )
        self.pos_embedding: torch.Tensor = nn.Parameter(torch.randn(self.n_patches + 1, 1, embedding_dim))
        self.cls_token: torch.Tensor = nn.Parameter(torch.randn(1, 1, embedding_dim))

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

        # Noise
        self.noise: Normal = Normal(0, noise)

        # Capsule layer
        self.caps_weights: torch.Tensor = nn.Parameter(0.01 * torch.randn(self.n_patches * caps_per_patch, caps_dim, embedding_dim))

        # Decoder
        self.decoder: nn.Sequential = nn.Sequential(nn.ConvTranspose2d(8, 16, 3, stride = 2, padding = 1),  # b,8,8,8 -> b,16,15,15
                                                    nn.BatchNorm2d(16),
                                                    nn.ReLU(inplace = True),
                                                    nn.ConvTranspose2d(16, 32, 9, stride = 3, padding = 1), # -> b,32,49,49
                                                    nn.BatchNorm2d(32),
                                                    nn.ReLU(inplace = True),
                                                    nn.ConvTranspose2d(32, 32, 7, stride = 5, padding = 1), # -> b,32,245,245
                                                    nn.BatchNorm2d(32),
                                                    nn.ReLU(inplace = True),
                                                    nn.ConvTranspose2d(32, 16, 9, stride = 2),              # -> b,16,497,497
                                                    nn.BatchNorm2d(16),
                                                    nn.ReLU(inplace = True),
                                                    nn.ConvTranspose2d(16, 8, 6, stride = 1),               # -> b,8,502,502
                                                    nn.BatchNorm2d(8),
                                                    nn.ReLU(inplace = True),
                                                    nn.ConvTranspose2d(8, 3, 11, stride = 1),               # -> b,3,512,512
                                                    nn.Tanh()
                                                    )

        # Losses
        self.mse_loss: nn.MSELoss = nn.MSELoss()
        self.ssim_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda x, y: 1 - ssim(x, y)  # type: ignore

        # MDN network
        self.mdn_weights: nn.Sequential = nn.Sequential(nn.Linear(embedding_dim, coefs, bias = False),
                                                        nn.Softmax(dim = -1)
                                                        )
        self.mdn_means: nn.Linear = nn.Linear(embedding_dim, embedding_dim * coefs, bias = False)
        self.mdn_logvars: nn.Sequential = nn.Sequential(nn.Linear(embedding_dim, embedding_dim * coefs, bias = False),
                                                        nn.Softplus()
                                                        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size: int = x.size(0)

        # ViT embedding
        embeds: torch.Tensor = self.embed_image(x)
        embeds = einops.rearrange(embeds, 'b d h w -> (h w) b d')

        # Add CLS and position
        cls_tokens: torch.Tensor = self.cls_token.expand(1, *embeds.size()[1:])
        seq: torch.Tensor = torch.cat((cls_tokens, embeds)) + self.pos_embedding

        # Transformer
        seq = self.transformer(seq)
        features: torch.Tensor = seq[1:]  # remove CLS token

        # Add noise during training
        if self.training:
            features += self.noise.sample(features.size()).to(features.device)

        # Capsule
        x_proj: torch.Tensor = torch.einsum('n c e, p b e -> b n e', self.caps_weights, features)
        x_proj_detached: torch.Tensor = x_proj.detach()
        logits: torch.Tensor = torch.zeros(batch_size,
                                           1,
                                           self.n_patches * self.hparams.caps_per_patch,    # type: ignore
                                           device = features.device
                                           )
        coeffs: torch.Tensor = F.softmax(logits, dim = -1)

        n_iters: int = self.hparams.caps_depth  # type: ignore
        capsule_out: torch.Tensor
        for _ in range(n_iters - 1):
            capsule_out = coeffs @ x_proj_detached
            capsule_out = self._squash(capsule_out)
            logits += capsule_out @ x_proj_detached.mT
            coeffs: torch.Tensor = F.softmax(logits, dim = -1)
        capsule_out = coeffs @ x_proj
        capsule_out = self._squash(capsule_out)

        # Decode
        caps_features: torch.Tensor = capsule_out.view(batch_size, 8, 8, 8)
        recon: torch.Tensor = self.decoder(caps_features)

        return features, recon
    
    @staticmethod
    def _squash(inputs: torch.Tensor) -> torch.Tensor:
        """
        Squashing function to normalize the capsule outputs.
        """

        norm: torch.Tensor = torch.norm(inputs, dim = -1, keepdim = True)
        scale: torch.Tensor = norm / (1 + norm ** 2)
        return scale * inputs


    def mdn_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the Mixture Density Network using the features from the encoder.
        """

        # Parameters
        logvars: torch.Tensor = self.mdn_logvars(features)
        means: torch.Tensor = self.mdn_means(features)
        weights: torch.Tensor = self.mdn_weights(features)

        # Reshape
        logvars = logvars.view(*features.size(), self.hparams.coefs)    # type: ignore
        means = means.view(*features.size(), self.hparams.coefs)    # type: ignore
        features = features.unsqueeze(-1)

        # Loss
        stds: torch.Tensor = torch.exp(logvars / 2)
        log_prob: torch.Tensor = Normal(means, stds).log_prob(features)
        log_prob = log_prob.sum(2)
        nll: torch.Tensor = -torch.einsum('p b e, p b e -> p b', log_prob, weights)

        return nll


    def loss_step(self, x: torch.Tensor) -> torch.Tensor:

        features, recon = self(x)

        # Loss terms
        loss_recon: torch.Tensor = self.mse_loss(recon, x)
        loss_ssim: torch.Tensor = self.ssim_loss(x, recon)
        loss_mdn: torch.Tensor = self.mdn_loss(features)
        loss_mdn = loss_mdn.sum(0).mean()

        # Combined loss
        loss: torch.Tensor = 5 * loss_recon + 0.5 * loss_ssim + loss_mdn

        return loss

    def training_step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        
        x: torch.Tensor = batch[0]
        loss: torch.Tensor = self.loss_step(x)
        self.log('train_loss', loss, prog_bar = True)
        return loss

    def validation_step(self, batch: list[torch.Tensor]) -> torch.Tensor:

        x: torch.Tensor = batch[0]
        loss: torch.Tensor = self.loss_step(x)
        self.log('val_loss', loss, prog_bar = True)
        return loss
    
    def predict_step(self, batch: list[torch.Tensor]) -> torch.Tensor:

        x: torch.Tensor = batch[0]
        features, recon = self(x)
        mdn_loss = self.mdn_loss(features)
        patch_size = self.hparams.patch_shape[0]   # type: ignore

        h = x.size(2) // patch_size
        w = x.size(3) // patch_size
        anomaly_map: torch.Tensor = mdn_loss.view(-1, 1, h, w)
        anomaly_map = torch.nn.UpsamplingBilinear2d((512, 512))(anomaly_map)
        # gaussian filter
        anomaly_map = gaussian_blur(anomaly_map, kernel_size = 33, sigma = 4) # type: ignore
        return anomaly_map
    
    def test_step(self, batch: list[torch.Tensor]) -> torch.Tensor:

        out: torch.Tensor = self.predict_step(batch)
        self.threshold = 0.5
        mask_out: torch.Tensor = out > self.threshold
        ground_truth: torch.Tensor = batch[1]
        auroc = binary_auroc(out, ground_truth)
        return auroc


    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(),
                                lr = self.hparams.lr,   # type: ignore
                                weight_decay = 1e-4
                                )
