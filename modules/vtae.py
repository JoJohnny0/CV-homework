"""
Module containing the VTAE class, implementing a Vision Transformer Autoencoder for anomaly detection.

Imports: network.capsule, network.decoder, network.encoder, network.mdn
"""

from collections.abc import Callable
from typing import Literal

from anomalib.metrics.pro import pro_score
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
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
                 loss_weights: tuple[float, float, float] = (1., 1. ,1.),
                 lr: float = 1e-3,
                 weight_decay: float = 0.,
                 use_dytanh: bool = False,
                 threshold: float = 0.5
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
            loss_weights: Weights for the reconstruction loss, SSIM loss, and MDN loss.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
            use_dytanh: Whether to use dyTanh normalization. If not, use layer normalization.
            threshold: Threshold for the anomaly map. Not used in the model, but can be used for evaluation.
        """

        super().__init__()
        self.save_hyperparameters()

        # Decoder
        self.decoder: decoder.Decoder = decoder.Decoder(out_shape = image_shape, in_channels = latent_channels)
        self.decoder_input_shape: tuple[int, int] = self.decoder.get_input_shape()

        n_patches: int = (image_shape[1] // patch_shape[0]) * (image_shape[2] // patch_shape[1])
        embedding_dim: int = latent_channels * self.decoder_input_shape[0] * self.decoder_input_shape[1]

        # ViT encoder
        self.encoder: encoder.ViTEncoder = encoder.ViTEncoder(patch_shape = patch_shape,
                                                              n_patches = n_patches,
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
        self.caps = capsule.Capsule(n_caps = n_patches * caps_per_patch,
                                    caps_dim = caps_dim,
                                    output_dim = embedding_dim,
                                    iterations = caps_iterations
                                    )
        
        # MDN
        self.mdn = mdn.MDN(input_dim = embedding_dim, n_components = mdn_components)

        # Losses
        self.mse_loss: nn.MSELoss = nn.MSELoss()
        self.ssim_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda x, y: 1 - ssim(x, y)  # type: ignore

        # Upsampler
        self.upsampler: nn.UpsamplingBilinear2d = nn.UpsamplingBilinear2d((image_shape[1], image_shape[2]))


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
        nll: torch.Tensor = -torch.einsum('p b e, p b e -> b p', log_prob, pi)

        return nll

    def loss_step(self, batch: list[torch.Tensor], split: Literal['train', 'val']) -> torch.Tensor:
        """
        Compute the total loss for a single step.
        """

        # Extract the input tensor
        x: torch.Tensor = batch[0]

        # Forward pass 
        features: torch.Tensor
        recon: torch.Tensor
        features, recon = self(x)

        # Save the recontruction to log at train epoch end
        self.latest_recon: torch.Tensor = recon

        # Loss terms
        recon_loss: torch.Tensor = self.mse_loss(recon, x)
        ssim_loss: torch.Tensor = self.ssim_loss(x, recon)
        mdn_loss: torch.Tensor = self.mdn_loss(features)
        mdn_loss = mdn_loss.sum(-1).mean()

        # Log the losses
        self.log(f'{split}_recon_loss', recon_loss, prog_bar = True)
        self.log(f'{split}_ssim_loss', ssim_loss, prog_bar = True)
        self.log(f'{split}_mdn_loss', mdn_loss, prog_bar = True)

        # Weight the losses
        recon_loss = self.hparams.loss_weights[0] * recon_loss  # type: ignore
        ssim_loss = self.hparams.loss_weights[1] * ssim_loss    # type: ignore
        mdn_loss = self.hparams.loss_weights[2] * mdn_loss  # type: ignore

        # Combined loss
        combined_loss: torch.Tensor = recon_loss + ssim_loss + mdn_loss
        self.log(f'{split}_combined_loss', combined_loss, prog_bar = True)

        return combined_loss

    def training_step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        return self.loss_step(batch, split = 'train')

    def validation_step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        return self.loss_step(batch, split = 'val')
    
    def on_train_epoch_end(self) -> None:

        # If using TensorBoard, log the latest reconstruction image
        if isinstance(self.logger, TensorBoardLogger):
            self.logger.experiment.add_image('reconstruction', self.latest_recon[0], self.current_epoch)
    

    def predict_step(self, batch: list[torch.Tensor]) -> torch.Tensor:

        x: torch.Tensor = batch[0]
        features: torch.Tensor = self(x)[0]
        mdn_loss: torch.Tensor = self.mdn_loss(features)
        
        # Normalize the MDN loss
        mdn_loss = (mdn_loss - mdn_loss.min()) / (mdn_loss.max() - mdn_loss.min() + 1e-8)

        # Number of patches on the vertical and horizontal axes
        vertical_patches: int = self.hparams.image_shape[1] // self.hparams.patch_shape[0]  # type: ignore
        horizontal_patches: int = self.hparams.image_shape[2] // self.hparams.patch_shape[1]    # type: ignore

        # Reshape the anomaly map
        anomaly_map: torch.Tensor = mdn_loss.view(x.size(0), 1, vertical_patches, horizontal_patches)

        # Upsample to original size
        anomaly_map = self.upsampler(anomaly_map)

        # gaussian filter
        sigma: list[float] = [vertical_patches / 2, horizontal_patches / 2]
        kernel_size: list[int] = [2 * round(4 * s) + 1 for s in sigma]
        anomaly_map = gaussian_blur(anomaly_map, kernel_size = kernel_size, sigma = sigma)
        
        return anomaly_map
    
    def test_step(self, batch: list[torch.Tensor]) -> None:

        predictions: torch.Tensor = self.predict_step(batch)
        ground_truth: torch.Tensor = batch[1]

        # Compute the metrics
        auroc: torch.Tensor = binary_auroc(predictions, ground_truth.int())
        pro: torch.Tensor = pro_score(predictions.float(), ground_truth, threshold = self.hparams.threshold)    # type: ignore

        # Log the metrics
        self.log('auroc', auroc, prog_bar = True)
        self.log('pro_score', pro, prog_bar = True)

        return None


    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(),
                                lr = self.hparams.lr,   # type: ignore
                                weight_decay = self.hparams.weight_decay    # type: ignore
                                )
