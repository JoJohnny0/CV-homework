"""
Module containing the VTAE class, implementing a Vision Transformer Autoencoder for anomaly detection.
"""

from collections.abc import Callable

import lightning.pytorch as pl
import torch
from torch.distributions import Normal
import torch.nn as nn
from torchmetrics.functional.classification import binary_auroc
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from torchvision.transforms.functional import gaussian_blur

from modules.network import capsule, decoder, encoder, mdn


class VTAE(pl.LightningModule):
    """
    Vision Transformer Autoencoder (VTAE) model for anomaly detection.
    """

    def __init__(self,
                 image_shape: tuple[int, int, int],
                 patch_shape: tuple[int, int],
                 latent_channels: int,
                 heads: int,
                 depth: int,
                 caps_per_patch: int,
                 caps_dim: int,
                 caps_iterations: int,
                 ff_dim: int,
                 mdn_components: int,
                 noise: float = 0.,
                 lr: float = 1e-3,
                 use_dytanh: bool = False
                 ) -> None:
        """
        Initialize the VTAE model.

        Args:
            image_shape: Shape of the input image (c, h, w).
            patch_shape: Shape of the patches used in the ViT encoder.
            latent_channels: Number of channels in the latent space.
            heads: Number of attention heads in the transformer encoder. To ensure compatibility with the decoder, it should be a divisor of the latent_channels. It may also work with other values.
            depth: Number of transformer encoder layers.
            caps_per_patch: Number of capsules per patch.
            caps_dim: Dimension of the capsules in the capsule layer.
            caps_iterations: Number of routing iterations in the capsule layer.
            ff_dim: Dimension of the feedforward network in the transformer encoder.
            mdn_components: Number of mixture components in the mixture density network.
            noise: Standard deviation of the Gaussian noise added to the features during training.
            lr: Learning rate for the optimizer.
            use_dytanh: Whether to use dyTanh normalization. If not, use layer normalization.
        """

        super().__init__()
        self.save_hyperparameters()

        # Decoder
        self.decoder: decoder.Decoder = decoder.Decoder(out_shape = image_shape, in_channels = latent_channels)
        self.decoder_input_shape: tuple[int, int] = self.decoder.get_input_shape()

        self.n_patches: int = (image_shape[1] // patch_shape[0]) * (image_shape[2] // patch_shape[1])
        embedding_dim: int = latent_channels * self.decoder_input_shape[0] * self.decoder_input_shape[1]

        # ViT encoder
        self.encoder: encoder.ViTEncoder = encoder.ViTEncoder(patch_shape = patch_shape,
                                                              n_patches = self.n_patches,
                                                              n_channels = image_shape[0],
                                                              embedding_dim = embedding_dim,
                                                              heads = heads,
                                                              depth = depth,
                                                              ff_dim = ff_dim,
                                                              use_dytanh = use_dytanh
                                                              )

        # Noise
        self.noise: Normal|None = Normal(0, noise) if noise > 0 else None

        # Capsule layer
        self.caps = capsule.Capsule(n_caps = self.n_patches * caps_per_patch,
                                    caps_dim = caps_dim,
                                    output_dim = embedding_dim,
                                    iterations = caps_iterations
                                    )

        # Losses
        self.mse_loss: nn.MSELoss = nn.MSELoss()
        self.ssim_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda x, y: 1 - ssim(x, y)  # type: ignore

        # MDN
        self.mdn = mdn.MDN(input_dim = embedding_dim, n_components = mdn_components)


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # Encode the features
        features: torch.Tensor = self.encoder(x)

        # Add noise during training
        if self.training and self.noise is not None:
            features += self.noise.sample(features.size()).to(features.device)

        # Capsule
        routed_caps: torch.Tensor = self.caps(features)
        routed_caps = routed_caps.view(x.size(0),
                                       self.hparams.latent_channels,    # type: ignore
                                       self.decoder_input_shape[0],
                                       self.decoder_input_shape[1]
                                       )

        # Decode
        recon: torch.Tensor = self.decoder(routed_caps)

        return features, recon


    def mdn_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the Mixture Density Network using the features from the encoder.
        """

        # MDN forward
        pi: torch.Tensor
        mu: torch.Tensor
        sigma: torch.Tensor
        pi, mu, sigma = self.mdn(features)

        # Loss
        stds: torch.Tensor = torch.exp(sigma / 2)
        log_prob: torch.Tensor = Normal(mu, stds).log_prob(features.unsqueeze(-1))
        log_prob = log_prob.sum(2)
        nll: torch.Tensor = -torch.einsum('p b e, p b e -> p b', log_prob, pi)

        return nll

    def loss_step(self, x: torch.Tensor) -> torch.Tensor:

        features: torch.Tensor
        recon: torch.Tensor
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
