from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from modules import mvtech
from modules.VT_AE import VTAE

# Data
train_loader, val_loader, test_loader = mvtech.get_loaders(product = 'hazelnut', batch_size = 8)

# Model
model = VTAE(patch_shape = (32, 16),
             embedding_dim = 256,
             depth = 7,
             coefs = 11,
             heads = 4,
             ff_dim = 1023,
             noise = 0.2,
             lr = 1e-4,
             use_dytanh = False,
             )

# Training
trainer: Trainer = Trainer(max_epochs = 1)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

import matplotlib.pyplot as plt
# ...existing code...

# Test
out_masks = trainer.predict(model, dataloaders=test_loader)
'''
for i, batch in enumerate(out_masks):
    for j, mask in enumerate(batch):
        # Assuming mask is a tensor, convert it to numpy for saving
        mask_np = mask.squeeze(0).detach().cpu().numpy()
        plt.imsave(f"prova/mask_{i}_{j}.png", mask_np, cmap='gray')'''
