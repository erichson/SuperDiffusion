"""
Created on Fri Jul  23, 2024

@author: ben
"""

import os, time
import numpy as np
import wandb
import torch
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier, get_rank

from diffusers.optimization import get_linear_schedule_with_warmup as scheduler


from src.unet import UNet
from src.diffusion_model import GaussianDiffusionModelSR
from src.get_data import NSTK_SR as NSTK
from src.plotting import plot_samples


def ddp_setup(local_rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "3522"
        init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
        rank = local_rank
    else:
        init_process_group(backend="nccl", 
                           init_method='env://')
        #overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()
        
    torch.cuda.set_device(local_rank)
    return local_rank, rank

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        local_gpu_id: int,
        sampling_freq: int,
        run: wandb,
        run_name: str
    ) -> None:
        self.gpu_id = gpu_id
        self.local_gpu_id = local_gpu_id
        self.model = model.to(self.local_gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.sampling_freq = sampling_freq
        self.model = DDP(model, device_ids=[self.local_gpu_id])
        self.run = run
        self.run_name = run_name

    def _run_batch(self, targets, conditioning_snapshots):
        self.optimizer.zero_grad()
        loss = self.model(targets, conditioning_snapshots)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.model.module.ema.update()
        loss_value = loss.item()
        return loss_value
        

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        loss_values = []
        for conditioning_snapshots, targets in self.train_data:
            conditioning_snapshots = conditioning_snapshots.to(self.local_gpu_id)
            targets = targets.to(self.local_gpu_id)            
            loss_values.append(self._run_batch(targets, conditioning_snapshots))
        return loss_values

    def _save_checkpoint(self, epoch):
        save_dict = {
            'model': self.model.module.eps_model.state_dict(),
            'ema': self.model.module.ema.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")
        PATH = "./checkpoints/" + "checkpoint_" + self.run_name + ".pt"
        torch.save(save_dict, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    
    def _generate_samples(self, epoch):
        with self.model.module.ema.average_parameters():
            self.model.eval()
            with torch.no_grad():
                PATH = "./train_samples_" + self.run_name
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
            
            
                self.train_data.sampler.set_epoch(1)
                conditioning_snapshots, targets = next(iter(self.train_data))
                conditioning_snapshots = conditioning_snapshots.to('cuda')
            
                samples = self.model.module.sample(conditioning_snapshots.shape[0], 
                                                   (1, targets.shape[2], targets.shape[3]), 
                                                   conditioning_snapshots, 'cuda')
                     
            
        plot_samples(samples, conditioning_snapshots, targets, PATH, epoch)
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
            
            start = time.time()
            loss_values.append(self._run_epoch(epoch))
            
            if self.gpu_id == 0:
                #Note that this average is not taken across all GPUs when multi-node is on and represents only the output of a single GPU
                avg_loss = np.mean(loss_values)
                
                print(f"Epoch {epoch} | loss {avg_loss} | learning rate {self.lr_scheduler.get_last_lr()}")
                print('Time taken for epoch {} is {} sec'.format(epoch, time.time()-start))
                
                if self.run is not None:
                    self.run.log({"loss": avg_loss})

            
            if self.gpu_id == 0 and epoch == 0:
                self._save_checkpoint(epoch+1)
                self._generate_samples(epoch+1)
                            
            if self.gpu_id == 0 and (epoch + 1) % self.sampling_freq == 0:
                self._save_checkpoint(epoch+1)
                self._generate_samples(epoch+1)

def load_train_objs(superres, args):
    #train_set = NSTK(path='data/16000_2048_2048_seed_3407_w.h5', factor=args.factor)
    train_set = NSTK(path='/pscratch/sd/v/vmikuni/FM/nskt_tensor/16000_2048_2048_seed_3407_w.h5', factor=args.factor)
    
    unet_model = UNet(image_size=256, in_channels=1, out_channels=1, superres=args.superres) 
    model = GaussianDiffusionModelSR(eps_model=unet_model.cuda(), betas=(1e-4, 0.02),
                                   n_T=args.time_steps, prediction_type = args.prediction_type, sampler = args.sampler)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        sampler=DistributedSampler(dataset),
        #num_workers=4
    )


def main(rank: int, world_size: int, sampling_freq: int, epochs: int, batch_size: int, run, args):
    
    #print(rank)
    #print(world_size)
    
    local_rank, rank = ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(superres=args.superres, args=args)
    

    if rank == 0:
        #==============================================================================
        # Model summary
        #==============================================================================
        print('**** Setup ****')
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')
        #    print(model)

    
    train_data = prepare_dataloader(dataset, batch_size)
    
    trainer = Trainer(model, train_data, optimizer, rank, local_rank, sampling_freq, run, run_name=args.run_name)
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
    parser.add_argument("--multi-node", action='store_true', default=False, help='Use multi node training')
    parser.add_argument('--factor', default=4, type=int, help='upsampling factor')

    parser.add_argument('--learning-rate', default=2e-4, type=int, help='learning rate')

    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")    
    parser.add_argument("--time-steps", type=int, default=1000, help="Time steps for sampling")    

    args = parser.parse_args()

    # Launch processes.
    print('Launching processes...')
    

    if args.multi_node:
        def is_master_node():
            return int(os.environ['RANK']) == 0
        
        if is_master_node():
            mode = None
            wandb.login()
        else:
            mode = "disabled"
            

        run = wandb.init(
            # Set the project where this run will be logged
            project="DiffusionSR",
            name=args.run_name,
            mode = mode,
            # Track hyperparameters and run metadata            
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch size": args.batch_size,
                "upsampling factor": args.factor,
            },
        )
            
        
        main(0,1,args.sampling_freq, args.epochs, args.batch_size, run, args)
    else:
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
