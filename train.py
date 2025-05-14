from lightning.pytorch import Trainer
from mvtech import Mvtec
from VT_AE import VTAE

# Data
dataset = Mvtec(batch_size = 8, product = 'hazelnut')
train_loader = dataset.train_loader
val_loader = dataset.validation_loader

# Model
model = VTAE(image_size = 512,
             patch_size = 64,
             embedding_dim = 512,
             depth = 6,
             coefs = 10,
             heads = 8,
             mlp_dim = 1024,
             channels = 3,
             lr = 1e-4
             )

# Training
trainer: Trainer = Trainer(max_epochs = 400)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

'''# Saving the best model
if np.mean(t_loss) <= minloss:
    minloss = np.mean(t_loss)
    ep = i
    os.makedirs('./saved_model', exist_ok=True)
    torch.save(model.state_dict(), f'./saved_model/VT_AE_Mvtech_{product}'+'.pt')
    torch.save(G_estimate.state_dict(), f'./saved_model/G_estimate_Mvtech_{product}'+'.pt')'''
