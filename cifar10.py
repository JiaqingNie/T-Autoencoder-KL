# %%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from torchvision.utils import make_grid
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from time import time
from modules import *
from autoencoder import TAutoencoderKL


# %%
def denorm(img_tensors, mean, std):
    # denormalize image tensors with mean and std of training dataset for all channels
    img_tensors = img_tensors.permute(1, 0, 2, 3)
    for t, m, s in zip(img_tensors, mean, std):
        t.mul_(s).add_(m)
    img_tensors = img_tensors.permute(1, 0, 2, 3)
    return img_tensors
    
    
def save_image(img, path, name):
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1,2,0)))
    # save image
    if not os.path.exists(path):
        os.mkdir(path)
    plt.savefig(f'{path}/{name}.png')

# %%
data_path = './data'
batch_size = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dat = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
test_dat = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

loader_train = DataLoader(train_dat, batch_size, shuffle=True)
loader_test = DataLoader(test_dat, batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# %%
latent_dim = (8,8,4)
model = TAutoencoderKL(latent_dim=latent_dim, image_size=32, patch_size=2, in_channels=3, hidden_size=768, depth=12, num_heads=6, mlp_ratio=6.0, num_classes=10, dropout_prob=0.1)
model.to(device)
opt_ae, opt_disc = model.configure_optimizers()

# %%
def train_VAE(model, device, loader_train, optimizer, num_epochs, latent_dim, beta_warm_up_period=1):
    assert isinstance(latent_dim, tuple) and len(latent_dim) == 3, "latent_dim must be a tuple of length 3"
    timestamp = int(time())
    
    opt_ae, opt_disc = optimizer
    
    train_losses = []
    test_losses = []
    global_step = 0
    for epoch in range(num_epochs):  
        model.train()
        train_loss = 0
        train_mse_loss = 0
        train_KL_div = 0
        
        beta = 1
        
        with tqdm.tqdm(loader_train, unit="batch") as tepoch: 
            for batch_idx, (data, y) in enumerate(tepoch):   
                data = data.to(device)
                y = y.to(device)
                
                opt_ae.zero_grad()
                opt_disc.zero_grad()
                
                data = model.get_input(data)
                reconstructions, posterior = model(data, y)
                
                aeloss, _ = model.loss(data, reconstructions, posterior, 0, global_step,
                                            last_layer=model.get_last_layer(), split="train")
                
                aeloss.backward()
                
                discloss, _ = model.loss(data, reconstructions, posterior, 1, global_step,
                                                last_layer=model.get_last_layer(), split="train")
                discloss.backward()
                
                opt_ae.step()
                opt_disc.step()
                
                tepoch.set_description(f"Epoch {epoch}, Global step {global_step}")
                tepoch.set_postfix(aeloss=aeloss.item()/len(data), discloss=discloss.item()/len(data))
                global_step += 1
    
            
            model.eval()
            with torch.no_grad():
                data, y = next(iter(loader_test))
                data = model.get_input(data)
                data = data.to(device)
                y = y.to(device)
                reconstructions, posterior = model(data, y)
                reconstructions = reconstructions.cpu()
                recons = make_grid(torch.clamp(denorm(reconstructions, mean, std), 0., 1.), nrow=4, padding=0, normalize=False,
                                        range=None, scale_each=False, pad_value=0)
                plt.figure(figsize = (8,8))
                save_image(recons, f'./T-Autoencoder-KL/results/{timestamp}', epoch+1)

        # save the model
        if epoch == num_epochs - 1:
            with torch.no_grad():
                torch.save(model, f'./T-Autoencoder-KL/results/{timestamp}/TVQVAE.pt')
    return train_losses


# %%
if __name__ == '__main__':
    train_losses = train_VAE(model, device, loader_train, (opt_ae, opt_disc), 8, latent_dim, beta_warm_up_period=10)
