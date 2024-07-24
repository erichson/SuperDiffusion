"""
Created on Fri Jul  23, 2024

@author: ben
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import matplotlib.pyplot as plt
import cmocean

from src.unet import UNet
from src.ddpm import DDPM
from src.get_data import NSTK_SR as NSTK


# export CUDA_VISIBLE_DEVICES=1,3,5,6,7; python train_nstk.py


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        sampling_freq: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.sampling_freq = sampling_freq
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, targets, condining_snapshot):
        self.optimizer.zero_grad()
        loss = self.model(targets, condining_snapshot)
        loss.backward()
        loss_value = loss.item()
        self.optimizer.step()
        return loss_value
        

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        loss_values = []
        for condining_snapshot, targets in self.train_data:
            condining_snapshot = condining_snapshot.to(self.gpu_id)
            targets = targets.to(self.gpu_id)            
            loss_values.append(self._run_batch(targets, condining_snapshot))
        return loss_values

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    
    def _generate_samples(self, epoch):
        self.model.eval()
        with torch.no_grad():
            PATH = "./train_samples/"
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            
            
            self.train_data.sampler.set_epoch(1)
            source, target = next(iter(self.train_data))
            source = source.to('cuda')
            
            samples = self.model.module.sample(4, (1, 512, 512), source, 'cuda')
            
            nrow = 3; ncol = 4;
            f, axarr = plt.subplots(nrow, ncol, figsize=(12, 8))

            for i in range(ncol): 
               axarr[0,i].imshow(source[i,0,:,:].cpu().detach().numpy(), cmap=cmocean.cm.balance)
               axarr[0,i].set_xticks([])
               axarr[0,i].set_yticks([])
               axarr[0,i].title.set_text("LR input")  

            for i in range(ncol): 
               axarr[1,i].imshow(samples[i,0,:,:].cpu().detach().numpy(), cmap=cmocean.cm.balance)
               axarr[1,i].set_xticks([])
               axarr[1,i].set_yticks([])
               axarr[1,i].title.set_text("Super-resolved")  

            for i in range(ncol): 
               axarr[2,i].imshow(target[i,0,:,:].cpu().detach().numpy(), cmap=cmocean.cm.balance)
               axarr[2,i].set_xticks([])
               axarr[2,i].set_yticks([])
               axarr[2,i].title.set_text("HR Ground Truth")                 

            plt.tight_layout()
            plt.savefig(PATH + "ddpm_sample_"+ str(epoch) + ".png")
            plt.close()
        print(f"Epoch {epoch} | Generated samples saved at {PATH}")


    def train(self, max_epochs: int):
        
        self.model.train()
        for epoch in range(max_epochs):
            loss_values = []
            loss_values.append(self._run_epoch(epoch))
            
            if self.gpu_id == 0:
                avg_loss = np.mean(loss_values)
                print(f"Epoch {epoch} | loss {avg_loss}")
            
            if self.gpu_id == 0 and (epoch) % self.sampling_freq == 0:
                self._save_checkpoint(epoch)
                self._generate_samples(epoch)


def load_train_objs(superres, args):
    train_set = NSTK(path='16000_2048_2048_seed_3407_w.h5', factor=args.factor)  # load your dataset
    model = UNet(image_size=512, in_channels=1, out_channels=1, lowres_cond=args.superres) # load your model
    ddpm = DDPM(eps_model=model.cuda(), betas=(1e-4, 0.02), n_T=1000) # maybe use 0.03 ?
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4)
    return train_set, ddpm, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, sampling_freq: int, total_epochs: int, batch_size: int, args):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(superres=args.superres, args=args)
    
    
    #==============================================================================
    # Model summary
    #==============================================================================
    print('**** Setup ****')
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')
    #    print(model)

    
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, sampling_freq)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=1500, type=int, help='Total epochs to train the model')
    parser.add_argument('--sampling_freq', default=10, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')

    parser.add_argument('--superres', default=True, type=bool, help='Superresolution')
    parser.add_argument('--factor', default=4, type=int, help='upsampling factor')
    
    
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.sampling_freq, args.total_epochs, args.batch_size, args), nprocs=world_size)

