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
# from src.swinIR import SwinIR
from src.hat_arch import HAT
from diffusers.optimization import get_linear_schedule_with_warmup as scheduler

# from src.unet import UNet
# from src.diffusion_model import GaussianDiffusionModelSR
from src.get_data import NSTK_Cast as NSTK
from src.plotting import plot_samples


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "3522"
    init_process_group(backend="gloo", rank=rank, world_size=world_size)
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
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def _run_batch(self, targets, conditioning_snapshots):
        self.optimizer.zero_grad()
        outputs = self.model(conditioning_snapshots)
        targets = targets
        # print("***********!!!!!***********")   
        # print(targets.shape, outputs.shape, conditioning_snapshots.shape)  
        loss = self.criterion(outputs.flatten(),targets.flatten())
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        # self.model.module.ema.update()
        loss_value = loss.item()
        # print(f"Rank {self.gpu_id} | Loss {loss_value}")
        # if self.gpu_id == 0:
        #     self.run.log({"step_loss": loss_value})
        return loss_value
        

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        loss_values = []
        for conditioning_snapshots, targets, _, _ in self.train_data:
            conditioning_snapshots = conditioning_snapshots.to(self.gpu_id)
            targets = targets.to(self.gpu_id)      
            loss_values.append(self._run_batch(targets, conditioning_snapshots))
            if self.gpu_id == 0:
                print(f"progress = {len(loss_values)}/{len(self.train_data)} - mean_loss = {np.mean(loss_values)}")
        return loss_values

    def _save_checkpoint(self, epoch):
        save_dict = {
            'model': self.model.state_dict(),
            # 'ema': self.model.module.ema.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if not os.path.exists("./checkpoints"):
                os.makedirs("./checkpoints")
        PATH = "./checkpoints/" + "checkpoint_" + self.run_name + str(epoch) + ".pt"
        torch.save(save_dict, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    
    def _generate_samples(self, epoch):
        # with self.model.module.ema.average_parameters():
            self.model.eval()
            with torch.no_grad():
                PATH = "./train_samples_" + self.run_name
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
            
            
                self.train_data.sampler.set_epoch(1)
                conditioning_snapshots, targets, _, _ = next(iter(self.train_data))
                conditioning_snapshots = conditioning_snapshots.to('cuda')
            
                samples = self.model(conditioning_snapshots)
                     
            
            plot_samples(samples, conditioning_snapshots, targets, PATH, epoch)
            print(f"Epoch {epoch} | Generated samples saved at {PATH}")


    def train(self, max_epochs: int):
        
        self.lr_scheduler = scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=len(self.train_data) * 5, # we need only a very shot warmup phase for our data
            num_training_steps=(len(self.train_data) * max_epochs),
        )

        
        self.model.train()
        
        for epoch in range(max_epochs):
            loss_values = []
            loss_values.append(self._run_epoch(epoch))
            
            if self.gpu_id == 0:
                avg_loss = np.mean(loss_values)
                print(f"Epoch {epoch} | loss {avg_loss} | learning rate {self.lr_scheduler.get_last_lr()}")
                self.run.log({"epoch_loss": avg_loss})

            
            if self.gpu_id == 0 and epoch == 0:
                print(f"Init Epoch | Saving checkpoint and generating samples")
                self._save_checkpoint(epoch+1)
                self._generate_samples(epoch+1)

            
            if self.gpu_id == 0 and (epoch + 1) % self.sampling_freq == 0:
                print(f"Epoch {epoch+1} | Saving checkpoint and generating samples")
                self._save_checkpoint(epoch+1)
                self._generate_samples(epoch+1)


def load_train_objs(superres, args):
    #train_set = NSTK(path='data/16000_2048_2048_seed_3407_w.h5', factor=args.factor)
    train_set = NSTK(num_pred_steps=0, factor=args.factor)
    
    # unet_model = UNet(image_size=256, in_channels=1, out_channels=1, superres=args.superres) 
    # model = GaussianDiffusionModelSR(eps_model=unet_model.cuda(), betas=(1e-4, 0.02),
    #                                n_T=args.time_steps, prediction_type = args.prediction_type, sampler = args.sampler)
    # model = SwinIR(img_size=32, patch_size=1, in_chans=1,
    #                window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
    #                embed_dim=256, num_heads=[8,8,8,8,8,8], mlp_ratio=4, upsampler='pixelshuffle',
    #                upscale=args.factor,resi_connection='1conv') #img_range dose nothing here
    model = HAT(img_size=32, patch_size=1, in_chans=1,
                window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                embed_dim=256, num_heads=[8,8,8,8,8,8], mlp_ratio=4, upsampler='pixelshuffle',
                upscale=args.factor, resi_connection='1conv') #TODO
    
    # def print_layer_output_shape(module, input, output):
    #     print(f"Layer: {module.__class__.__name__}")
    #     print(f"  Input shape: {input[0].shape}")   # Input is a tuple, usually we care about the first element
    #     print(f"  Output shape: {output.shape}")
    #     print("-" * 40)
        
    # def register_hooks(model):
    #     for layer in model.children():  # Register hook for each layer
    #         layer.register_forward_hook(print_layer_output_shape)

    # register_hooks(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        #num_workers=4
    )


def main(rank: int, world_size: int, sampling_freq: int, epochs: int, batch_size: int, run, args):
    
     # Initialize W&B only on the main process (rank 0)
    run = run
    wandb.login()
    
    if rank == 0:
        run = wandb.init(
            project="HAT-SR-NERSC",
            name=args.run_name,
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch size": args.batch_size,
                "upsampling factor": args.factor,
            },
        )
    else:
        run = None
    
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(superres=args.superres, args=args)
    
    
    #==============================================================================
    # Model summary
    #==============================================================================
    print('**** Setup ****')
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')
    # print(model)

    
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, sampling_freq, run, run_name=args.run_name)
    
    print(f"Rank {rank} | Training started")
    trainer.train(epochs)
    destroy_process_group()
    if rank == 0:
        wandb.finish()  # Close W&B only in the main process


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument("--run-name", type=str, default='hat_sr_1', help="Name of the current run.")
    parser.add_argument('--epochs', default=500, type=int, help='Total epochs to train the model')
    parser.add_argument('--sampling-freq', default=5, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch-size', default=32, type=int, help='Input batch size on each device (default: 16)')

    parser.add_argument('--superres', default=True, type=bool, help='Superresolution')
    parser.add_argument('--factor', default=8, type=int, help='upsampling factor')

    parser.add_argument('--learning-rate', default=2e-4, type=int, help='learning rate')

    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")    
    parser.add_argument("--time-steps", type=int, default=1000, help="Time steps for sampling")    

    args = parser.parse_args()

    # Launch processes.
    print('Launching processes...')
    
    print(f"Run name: {args.run_name}, sampling freq: {args.sampling_freq}, epochs: {args.epochs}, batch size: {args.batch_size}")

    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.sampling_freq, args.epochs, args.batch_size, None, args), nprocs=world_size)
