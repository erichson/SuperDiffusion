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
from src.diffusion_model import GaussianDiffusionModel
from src.get_data import NSTK


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
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        loss = self.model(source)
        loss.backward()
        loss_value = loss.item()
        #pbar.set_description(f"epoch: {i:.1f}; loss: {loss_value:.4f}")
        self.optimizer.step()
        return loss_value
        

    def _run_epoch(self, epoch):
        #b_sz = len(next(iter(self.train_data))[0])
        #print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        loss_values = []
        for source in self.train_data:
            source = source.to(self.gpu_id)
            #targets = targets.to(self.gpu_id)
            loss_values.append(self._run_batch(source, None))
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
            
            samples = self.model.module.sample(6, (1, 512, 512), 'cuda')
            
            nrow = 2; ncol = 3;
            fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(10.5, 6.5), layout="constrained")
            axs = np.array(axs)
            for i, ax in enumerate(axs.reshape(-1)):
                ax.imshow(samples[i,0,:,:].cpu().detach().numpy(), cmap=cmocean.cm.balance)
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
            
            if self.gpu_id == 0 and (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch)
                self._generate_samples(epoch)


def load_train_objs():
    train_set = NSTK(path='16000_2048_2048_seed_3407_w.h5')  # load your dataset
    unet_model = UNet(image_size=512, in_channels=1, out_channels=1) # load your model
    model = GaussianDiffusionModel(eps_model=unet_model.cuda(), betas=(1e-4, 0.04), n_T=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs()
    
    
    #==============================================================================
    # Model summary
    #==============================================================================
    print('**** Setup ****')
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')
    #    print(model)

    
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', default=1500, type=int, help='Total epochs to train the model')
    parser.add_argument('--save_every', default=25, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)

