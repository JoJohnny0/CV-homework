import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCaps(nn.Module):

    def __init__(self, out_num_caps: int = 1, in_num_caps: int = 8*8*64, in_dim_caps: int = 8, out_dim_caps: int = 512) -> None:
        super().__init__()
        self.in_num_caps: int = in_num_caps
        self.out_num_caps: int = out_num_caps
        self.W: torch.Tensor = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, in_num_caps, in_dim_caps]
        x_hat: torch.Tensor = torch.einsum('oicd,bid->boic', self.W, x)  # [batch, out_num_caps, in_num_caps, out_dim_caps]
        logits: torch.Tensor = torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps, device = x.device)
        coeffs: torch.Tensor = F.softmax(logits, dim = 1)

        num_iters: int = 3
        capsule_out: torch.Tensor
        for _ in range(num_iters - 1):
            capsule_out = self.squash(torch.sum(coeffs.unsqueeze(-1) * x_hat.detach(), dim = 2))
            logits = logits + torch.sum(capsule_out.unsqueeze(2) * x_hat.detach(), dim = -1)
            coeffs: torch.Tensor = F.softmax(logits, dim = 1)
        capsule_out = self.squash(torch.sum(coeffs.unsqueeze(-1) * x_hat, dim = 2))

        # Masking
        norms: torch.Tensor = torch.norm(capsule_out, dim = -1)
        predictions: torch.Tensor = norms.argmax(dim = 1)
        mask: torch.Tensor = F.one_hot(predictions, num_classes = self.out_num_caps).float()
        masked_out: torch.Tensor = torch.sum(capsule_out * mask.unsqueeze(-1), dim = 1, keepdim = True)
        return masked_out

    @staticmethod
    def squash(inputs: torch.Tensor) -> torch.Tensor:
        norm: torch.Tensor = torch.norm(inputs, dim = -1, keepdim = True)
        scale: torch.Tensor = norm / (1 + norm ** 2)
        return scale * inputs
