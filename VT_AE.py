from collections.abc import Callable
from typing import Literal

import torch
import torch.nn as nn
import lightning.pytorch as pl
from einops import rearrange, repeat
from torch.distributions import Normal
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

import spatial as S
import mdn1


class CustomNormTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, norm_layer: Callable[..., nn.Module] = nn.LayerNorm, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        d_model: int = args[0] if len(args) > 0 else kwargs['d_model']
        self.norm1: nn.Module = norm_layer(d_model)
        self.norm2: nn.Module = norm_layer(d_model)


class VTAE(pl.LightningModule):

    def __init__(self,
                 image_size,
                 patch_size: int,
                 embedding_dim: int,
                 depth: int,
                 heads: int,
                 mlp_dim: int,
                 channels: int = 3,
                 lr: float = 1e-3
                 ) -> None:
        
        super().__init__()
        self.save_hyperparameters()

        # Vision Transformer encoder
        num_patches: int = (image_size // patch_size) ** 2
        patch_dim: int = channels * patch_size * patch_size

        self.patch_to_embedding: nn.Linear = nn.Linear(patch_dim, embedding_dim)
        self.pos_embedding: torch.Tensor = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))
        self.cls_token: torch.Tensor = nn.Parameter(torch.randn(1, 1, embedding_dim))

        encoder_layer: nn.TransformerEncoderLayer = CustomNormTransformerEncoderLayer(d_model = embedding_dim,
                                                                                      nhead = heads,
                                                                                      dim_feedforward = mlp_dim,
                                                                                      activation = 'gelu',
                                                                                      norm_first = True,
                                                                                      batch_first = True
                                                                                      )
        self.transformer: nn.TransformerEncoder = nn.TransformerEncoder(encoder_layer, num_layers = depth, enable_nested_tensor = False)

        # Capsule layer
        in_caps: int = num_patches * 8 * 8
        self.digcap: S.DigitCaps = S.DigitCaps(in_num_caps=in_caps, in_dim_caps = 8)

        # Decoder
        self.decoder: nn.Sequential = nn.Sequential(nn.ConvTranspose2d(8, 16, 3, stride = 2, padding = 1), # b,8,8,8 -> b,16,15,15
                                                    nn.BatchNorm2d(16),
                                                    nn.ReLU(),
                                                    nn.ConvTranspose2d(16, 32, 9, stride = 3, padding = 1),    # -> b,32,49,49
                                                    nn.BatchNorm2d(32),
                                                    nn.ReLU(),
                                                    nn.ConvTranspose2d(32, 32, 7, stride = 5, padding = 1),    # -> b,32,245,245
                                                    nn.BatchNorm2d(32),
                                                    nn.ReLU(),
                                                    nn.ConvTranspose2d(32, 16, 9, stride = 2), # -> b,16,497,497
                                                    nn.BatchNorm2d(16),
                                                    nn.ReLU(),
                                                    nn.ConvTranspose2d(16, 8, 6, stride = 1),    # -> b,8,502,502
                                                    nn.BatchNorm2d(8),
                                                    nn.ReLU(),
                                                    nn.ConvTranspose2d(8, 3, 11, stride = 1),    # -> b,3,512,512
                                                    nn.Tanh()
                                                    )

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Losses
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = lambda x, y: -torch.Tensor(ssim(x, y))
        self.mdn_loss = mdn1.mdn_loss_function

        # MDN network
        self.G_estimate = mdn1.MDN()


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size: int = x.size(0)

        # ViT embedding
        patches: torch.Tensor = rearrange(x,
                                          'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                                          p1 = self.hparams.patch_size, # type: ignore
                                          p2 = self.hparams.patch_size  # type: ignore
                                          )
        embeds: torch.Tensor = self.patch_to_embedding(patches)

        # Add CLS and position
        cls_tokens: torch.Tensor = repeat(self.cls_token, '() n d -> b n d', b = batch_size)
        seq: torch.Tensor = torch.cat((cls_tokens, embeds), dim = 1) + self.pos_embedding[:, :embeds.size(1) + 1]

        # Transformer
        seq = self.transformer(seq)

        # remove CLS
        features: torch.Tensor = seq[:, 1:, :]

        # Add noise during training
        if self.training:
            features += Normal(0, 0.2).sample(features.shape).to(features.device)

        # Capsule
        flat: torch.Tensor = features.view(batch_size, -1, 8)
        caps_output, _ = self.digcap(flat)

        # Decode
        caps_feat: torch.Tensor = caps_output.view(batch_size, -1, 8, 8)
        recon: torch.Tensor = self.decoder(caps_feat)

        return features, recon


    def step(self, batch: list[torch.Tensor], split: Literal['train', 'val']) -> torch.Tensor:

        x: torch.Tensor = batch[0]
        features, recon = self(x)

        # MDN forward
        pi, mu, sigma = self.G_estimate(features)

        # Loss terms
        loss_recon: torch.Tensor = self.mse_loss(recon, x)
        loss_ssim: torch.Tensor = self.ssim_loss(x, recon)
        loss_mdn: torch.Tensor = self.mdn_loss(features, mu, sigma, pi)

        # Combined loss
        loss: torch.Tensor = 5 * loss_recon + 0.5 * loss_ssim + loss_mdn
        self.log(f'{split}_loss', loss, prog_bar = True)

        return loss

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'train')

    def validation_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.step(batch, 'val')

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(list(self.parameters()) + list(self.G_estimate.parameters()),
                                lr = self.hparams.lr,   # type: ignore
                                weight_decay = 1e-4
                                )
