# %%
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torchvision.datasets import LSUN
from torchvision import datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import logging

from torchvision.utils import make_grid
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from time import time
from modules import *
from autoencoder import TAutoencoderKL

timestamp = int(time())

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
        os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/{name}.png')

class LSUNWithLabels:
    def __init__(self, lsun_classes, root_dir, transform=None, num_samples=50000):
        self.datasets = []
        
        for idx, cls in enumerate(lsun_classes):
            full_dataset = LSUN(root=root_dir, classes=[cls], transform=transform)
            
            def wrapped_getitem(index, dataset=full_dataset):
                data, _ = dataset[index]
                return data
            
            indices = torch.randperm(len(full_dataset))[:num_samples]
            subset = [(wrapped_getitem(index), idx) for index in indices]
            self.datasets.append(subset)
            
        self.lengths = [len(subset) for subset in self.datasets]
        self.total_length = sum(self.lengths)

    def __getitem__(self, index):
        for i, subset in enumerate(self.datasets):
            if index < self.lengths[i]:
                return subset[index]
            index -= self.lengths[i]
        raise IndexError

    def __len__(self):
        return self.total_length
# %%
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def setup_logger(rank):
    logger = logging.getLogger(f"Rank_{rank}")
    logger.setLevel(logging.INFO)
    filename = f'./logs/rank_{rank}.log'
    if os.path.exists(filename):
        os.remove(filename)
    fh = logging.FileHandler(filename)  # 每个 rank 有自己的日志文件
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger
    
def train(rank, world_size, num_epochs=8):
    try:
        setup(rank, world_size)
        logger = setup_logger(rank)
        
        data_path = '/prj/data/lsun'
        batch_size = 4
        img_size = 64
        latent_dim = (16,16,4)
        num_samples = 50000
        ckpt_period = 2
        disc_train_period = 1
        disc_start = 37501
        
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


        lsun_classes = ['bedroom_train', 'tower_train', 'bridge_train']
        val_classes = ['bedroom_val', 'tower_val', 'bridge_val']

        # Load the training dataset
        logger.info(f"Loading LSUN dataset of classes: {', '.join(lsun_classes)}...")
        train_dataset = LSUNWithLabels(lsun_classes, root_dir=data_path, transform=transform, num_samples=num_samples)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler)
        
        logger.info("Done!")
        
        # Load the validation dataset
        logger.info(f"Loading LSUN dataset of classes: {', '.join(val_classes)}...")
        val_dataset = LSUNWithLabels(val_classes, root_dir=data_path, transform=transform, num_samples=100)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        val_loader = DataLoader(val_dataset, batch_size, sampler=val_sampler)
        
        logger.info("Done!")

        # %%
        model = TAutoencoderKL(latent_dim=latent_dim, 
                               image_size=img_size, 
                               patch_size=2, 
                               in_channels=3, 
                               hidden_size=768, 
                               depth=12, 
                               num_heads=6, 
                               mlp_ratio=6.0, 
                               num_classes=3, 
                               dropout_prob=0.1, 
                               disc_start=disc_start,
                               kl_weight=1.0e-1,
                               num_epochs=num_epochs)
        
        model.to(rank)
        model = DistributedDataParallel(model, device_ids=[rank])
        logger.info(f"TAKL Parameters: {sum(p.numel() for p in model.parameters()):,}")
        opt_ae, opt_disc = model.module.configure_optimizers()
        
        log_interval = 10


        assert isinstance(latent_dim, tuple) and len(latent_dim) == 3, "latent_dim must be a tuple of length 3"
        
        model.eval()
        with torch.no_grad():
            data, y = next(iter(val_loader))
            data = model.module.get_input(data)
            data = data.to(rank)
            y = y.to(rank)
            recons = make_grid(torch.clamp(denorm(data, mean, std), 0., 1.), nrow=2, padding=0, normalize=False,
                                    range=None, scale_each=False, pad_value=0)
            plt.figure(figsize = (8,8))
            save_image(recons, f'./results/{timestamp}', f"0-{rank}")
        
        global_step = 0
        logger.info(f"Training on rank {rank}...")
        for epoch in range(num_epochs):  
            model.train()
            
            with tqdm.tqdm(train_loader, unit="batch") as tepoch: 
                for batch_idx, (data, y) in enumerate(tepoch):   
                    data = data.to(rank)
                    y = y.to(rank)
                    opt_ae.zero_grad()
                    opt_disc.zero_grad()
                    
                    data = model.module.get_input(data)
                    reconstructions, posterior = model(data, y)
                    
                    aeloss, log = model.module.loss(data, reconstructions, posterior, 0, global_step,
                                            last_layer=model.module.get_last_layer(), split="train")
                
                    aeloss.backward()
                    opt_ae.step()
                
                    discloss = torch.tensor(0.)
                    if global_step % disc_train_period == 0:
                        discloss, _ = model.module.loss(data, reconstructions, posterior, 1, global_step,
                                                    last_layer=model.module.get_last_layer(), split="train", curr_epoch=epoch)
                        discloss.backward()
                        opt_disc.step()
                    
                    
                    if batch_idx % log_interval == 0 or batch_idx == len(train_loader)-1:
                        loss_text = f"Epoch {epoch} - Batch {batch_idx}"
                        for k in log:
                            loss_text += f" - {k}: {log[k].item()}"
                        if global_step >= disc_start:
                            loss_text += f" - disc_loss: {discloss.item()/len(data)}"
                        logger.info(loss_text)
                    global_step += 1
        
                model.eval()
                with torch.no_grad():
                    data, y = next(iter(val_loader))
                    data = model.module.get_input(data)
                    data = data.to(rank)
                    y = y.to(rank)
                    recon_x, _ = model(data, y)
                    recon_x = recon_x.cpu()
                    recons = make_grid(torch.clamp(denorm(recon_x, mean, std), 0., 1.), nrow=2, padding=0, normalize=False,
                                            range=None, scale_each=False, pad_value=0)
                    plt.figure(figsize = (8,8))
                    save_image(recons, f'./results/{timestamp}', f"{epoch+1}-{rank}")

            # save the model
            if (epoch + 1) % ckpt_period == 0 or epoch == num_epochs - 1:
                with torch.no_grad():
                    torch.save(model, f'./results/{timestamp}/TAKL-{epoch}.pt')

        return 
    
    except Exception as e:
        print(f"Error on rank {rank}: {e}")
    finally:
        cleanup()

def main():
    world_size = 4
    epochs = 10
    mp.spawn(train, args=(world_size, epochs), nprocs=world_size, join=True)

# %%
if __name__ == '__main__':
    main()