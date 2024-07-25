"""
Created on Fri Jul  23, 2024

@author: ben
"""

import os
import numpy as np
import wandb
import torch
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from diffusers.optimization import get_linear_schedule_with_warmup as scheduler

import matplotlib.pyplot as plt
import cmocean

from src.unet import UNet
from src.diffusion_model import GaussianDiffusionModel
from src.get_data import NSTK_SR as NSTK


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "33421"
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
        run: wandb,
        run_name: str
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.sampling_freq = sampling_freq
        self.model = DDP(model, device_ids=[gpu_id])
        self.run = run
        self.run_name = run_name

    def _run_batch(self, targets, condining_snapshot):
        self.optimizer.zero_grad()
        loss = self.model(targets, condining_snapshot)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        loss_value = loss.item()
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
        if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")
        PATH = "./checkpoints/" + "checkpoint_" + self.run_name + ".pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    
    def _generate_samples(self, epoch):
        self.model.eval()
        with torch.no_grad():
            PATH = "./train_samples_" + self.run_name
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            
            
            self.train_data.sampler.set_epoch(1)
            source, target = next(iter(self.train_data))
            source = source.to('cuda')
            
            samples = self.model.module.sample(4, (1, 1024, 1024), source, 'cuda')
            
            nrow = 3; ncol = 4;
            f, axarr = plt.subplots(nrow, ncol, figsize=(12, 8))

            for i in range(ncol): 
               axarr[0,i].imshow(source[i,0,:,:].cpu().detach().numpy(), cmap=cmocean.cm.balance)
               axarr[0,i].set_xticks([])
               axarr[0,i].set_yticks([])
               axarr[0,i].title.set_text("LR input")  

            for i in range(ncol):
               img = samples[i,0,:,:].cpu().detach().numpy() 
               axarr[1,i].imshow(img, cmap=cmocean.cm.balance)
               axarr[1,i].set_xticks([])
               axarr[1,i].set_yticks([])
               error = np.linalg.norm(img - target[i,0,:,:].cpu().detach().numpy()) / np.linalg.norm(target[i,0,:,:].cpu().detach().numpy())
               axarr[1,i].title.set_text(f"Super-resolved (RFNE={error:.3f})")  

            for i in range(ncol): 
               axarr[2,i].imshow(target[i,0,:,:].cpu().detach().numpy(), cmap=cmocean.cm.balance)
               axarr[2,i].set_xticks([])
               axarr[2,i].set_yticks([])
               axarr[2,i].title.set_text("HR Ground Truth")                 

            plt.tight_layout()
            plt.savefig(PATH + "/ddpm_sample_"+ str(epoch) + ".png")
            plt.close()
        print(f"Epoch {epoch} | Generated samples saved at {PATH}")


    def train(self, max_epochs: int):
        
        self.lr_scheduler = scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=len(self.train_data) * 5, # we need only a very shot warmup phase for our data
            num_training_steps=(len(self.train_data) * max_epochs),
        )
        #print(len(self.train_data))

        
        self.model.train()
        for epoch in range(max_epochs):
            loss_values = []
            loss_values.append(self._run_epoch(epoch))
            
            if self.gpu_id == 0:
                avg_loss = np.mean(loss_values)
                print(f"Epoch {epoch} | loss {avg_loss} | learning rate {self.lr_scheduler.get_last_lr()}")
                self.run.log({"loss": avg_loss})

            
            if self.gpu_id == 0 and epoch == 0:
                self._save_checkpoint(epoch+1)
                self._generate_samples(epoch+1)

            
            if self.gpu_id == 0 and (epoch + 1) % self.sampling_freq == 0:
                self._save_checkpoint(epoch+1)
                self._generate_samples(epoch+1)


def load_train_objs(superres, args):
    train_set = NSTK(path='16000_2048_2048_seed_3407_w.h5', factor=args.factor)  # load your dataset
    unet_model = UNet(image_size=1024, in_channels=1, out_channels=1, lowres_cond=args.superres) # load your model
    model = GaussianDiffusionModel(eps_model=unet_model.cuda(), betas=(1e-4, 0.02), n_T=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, sampling_freq: int, epochs: int, batch_size: int, run, args):
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
    trainer = Trainer(model, train_data, optimizer, rank, sampling_freq, run, run_name=args.run_name)
    trainer.train(epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument("--run-name", type=str, default='run1', help="Name of the current run.")
    parser.add_argument('--epochs', default=500, type=int, help='Total epochs to train the model')
    parser.add_argument('--sampling-freq', default=50, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch-size', default=16, type=int, help='Input batch size on each device (default: 32)')

    parser.add_argument('--superres', default=True, type=bool, help='Superresolution')
    parser.add_argument('--factor', default=4, type=int, help='upsampling factor')

    parser.add_argument('--learning-rate', default=2e-4, type=int, help='learning rate')
    
    
    args = parser.parse_args()

    # Launch processes.
    print('Launching processes...')
    
    wandb.login()
    
    run = wandb.init(
        # Set the project where this run will be logged
        project="DiffusionSR",
        name=args.run_name,
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch size": args.batch_size,
            "upsampling factor": args.factor,
        },
    )
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.sampling_freq, args.epochs, args.batch_size, run, args), nprocs=world_size)

