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
from torch.distributed import init_process_group, destroy_process_group, barrier, get_rank, is_initialized, all_reduce, get_world_size
from torch_ema import ExponentialMovingAverage


from torch.optim import lr_scheduler

from unet import UNet
from diffusion_model import GaussianDiffusionModel
from get_data import NSKT, E5, Simple
from plotting import plot_samples
from lion import Lion
from diffusers.optimization import get_linear_schedule_with_warmup as scheduler

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
    torch.backends.cudnn.benchmark = True
    return local_rank, rank


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        local_gpu_id: int,
        sampling_freq: int,
        run: wandb,
        epochs: int,
        run_name: str,
        scratch_dir: str,
        ema_val = 0.9999,
        clip_value = 1.0,
        fine_tune = False,
        dataset = 'nskt'
    ) -> None:
        self.gpu_id = gpu_id
        self.local_gpu_id = local_gpu_id
        self.model = model.to(local_gpu_id)
        self.ema = ExponentialMovingAverage(model.parameters(), decay=ema_val)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.clip_value = clip_value
        self.sampling_freq = sampling_freq
        self.model = DDP(model, device_ids=[local_gpu_id],
                         #find_unused_parameters=True
                         )
        self.run = run
        self.run_name = run_name
        self.fine_tune = fine_tune
        if self.fine_tune:
            self.run_name += '_fine_tune'
        self.dataset = dataset
        self.logs = {}
        self.startEpoch = 0
        self.best_loss = np.inf
        self.max_epochs = epochs
        self.checkpoint_dir = os.path.join(scratch_dir,'checkpoints')
        self.checkpoint_path = os.path.join(self.checkpoint_dir,f"checkpoint_{self.dataset}_{self.run_name}.pt")
        self.lr_scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=self.max_epochs)
                
        if os.path.isfile(self.checkpoint_path):
            if self.gpu_id ==0: print(f"Loading checkpoint from {self.checkpoint_path}")
            self._restore_checkpoint(self.checkpoint_path)
        elif self.fine_tune:
            pretrained_checkpoint = self.checkpoint_path.replace(f"{self.dataset}","nskt").replace("_fine_tune","")
            if self.gpu_id ==0: print(f"Loading pretrained checkpoint from {pretrained_checkpoint}")
            self._restore_checkpoint(pretrained_checkpoint,restore_all=False)

    def train_one_epoch(self):
        tr_time = 0
        self.model.train()
        # buffers for logs
        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.local_gpu_id)
        self.logs['train_loss'] = logs_buff[0].view(-1)

        tr_start = time.time()
        for lowres_snapshots, snapshots, future_snapshots, s, Reynolds_number in self.train_data:
            data_start = time.time()
            lowres_snapshots = lowres_snapshots.to(self.local_gpu_id)
            future_snapshots = future_snapshots.to(self.local_gpu_id)
            s = s.to(self.local_gpu_id)
            Reynolds_number = Reynolds_number.to(self.local_gpu_id)
            snapshots = snapshots.to(self.local_gpu_id)
            
            self.optimizer.zero_grad(set_to_none=True)

            loss = self.model(lowres_snapshots,snapshots, future_snapshots, s, Reynolds_number)
            
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
            self.optimizer.step()
            
            self.ema.update()
 
            # add all the minibatch losses
            self.logs['train_loss'] += loss.detach()

            tr_time += time.time() - tr_start
            
        self.logs['train_loss'] /= len(self.train_data)

        logs_to_reduce = ['train_loss']
        if is_initialized(): # reduce the logs across multiple GPUs
            for key in logs_to_reduce:
                all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/get_world_size())
                
        return tr_time



    def val_one_epoch(self):
        val_time = time.time()
        self.model.eval()
        
        # buffers for logs
        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.local_gpu_id)
        self.logs['val_loss'] = logs_buff[0].view(-1)

        with torch.no_grad():
            for lowres_snapshots, snapshots, future_snapshots, s, Reynolds_number in self.val_data:
                data_start = time.time()
                lowres_snapshots = lowres_snapshots.to(self.local_gpu_id)
                future_snapshots = future_snapshots.to(self.local_gpu_id)
                s = s.to(self.local_gpu_id)
                Reynolds_number = Reynolds_number.to(self.local_gpu_id)
                snapshots = snapshots.to(self.local_gpu_id)
                
                loss = self.model(lowres_snapshots,snapshots, future_snapshots, s, Reynolds_number)
                        
                # add all the minibatch losses
                self.logs['val_loss'] += loss.detach()
            
        self.logs['val_loss'] /= len(self.val_data)

        logs_to_reduce = ['val_loss']
        if is_initialized(): # reduce the logs across multiple GPUs
            for key in logs_to_reduce:
                all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/get_world_size())
                
        return time.time() - val_time


    def _save_checkpoint(self, epoch,PATH):
        save_dict = {
            'basemodel': self.model.module.encoder_model.state_dict(),
            'lowres_model': self.model.module.lowres_model.state_dict(),
            'forecast_model': self.model.module.forecast_model.state_dict(),
            'decoder_model': self.model.module.decoder_model.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch':epoch,
            'loss':self.best_loss,
            'sched': self.lr_scheduler.state_dict(),
        }

        if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
        
        torch.save(save_dict, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _restore_checkpoint(self,PATH,restore_all = True):
        checkpoint = torch.load(PATH, map_location='cuda:{}'.format(self.local_gpu_id))

        self.model.module.encoder_model.load_state_dict(checkpoint["basemodel"])
        self.model.module.lowres_model.load_state_dict(checkpoint["lowres_model"])
        self.model.module.forecast_model.load_state_dict(checkpoint["forecast_model"])
        self.model.module.decoder_model.load_state_dict(checkpoint["decoder_model"])
        self.ema.load_state_dict(checkpoint["ema"])

        if restore_all:
            self.startEpoch = checkpoint['epoch'] + 1
            if 'loss' in checkpoint:
                self.best_loss = checkpoint['loss']                
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'sched' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['sched'])

    
    def _generate_samples(self, epoch):
        with self.ema.average_parameters():
            self.model.eval()
            with torch.no_grad():
                PATH = "./train_samples_" + self.run_name
                if not os.path.exists(PATH):
                    os.makedirs(PATH)
                                
                self.val_data.sampler.set_epoch(1)
                lowres_snapshots, snapshots, future_snapshots, s, Reynolds_number = next(iter(self.val_data))
                snapshots = snapshots.to('cuda')
                lowres_snapshots = lowres_snapshots.to('cuda')
                s = s.to('cuda')
                Reynolds_number = Reynolds_number.to('cuda')

                
                samples = self.model.module.sample(snapshots.shape[0],
                                                   (1, snapshots.shape[2], snapshots.shape[3]), 
                                                   lowres_snapshots, s, Reynolds_number,'cuda')
                     
            
        plot_samples(samples, lowres_snapshots, snapshots[:,-1,None], PATH, epoch)
        print(f"Epoch {epoch} | Generated samples saved at {PATH}")


    def train(self):
        for epoch in range(self.startEpoch,self.max_epochs):
            if is_initialized():
                self.train_data.sampler.set_epoch(epoch)
            
            start = time.time()
            tr_time  = self.train_one_epoch()
            val_time = self.val_one_epoch()            
            self.lr_scheduler.step()

                                                
            if self.gpu_id == 0: 
                print("Epoch {} | loss {} | val loss {} | learning rate {}".format(epoch,self.logs['train_loss'],self.logs['val_loss'],self.lr_scheduler.get_last_lr()))
                print('Time taken for epoch {} is {} sec'.format(epoch, time.time()-start))
                if self.run is not None:
                    self.run.log({"val_loss": self.logs['val_loss']})
                    self.run.log({"train_loss": self.logs['train_loss']})


            # if self.gpu_id == 0:                
                # if epoch == 0:
                #     self._save_checkpoint(epoch+1,self.checkpoint_path)
                #     #self._generate_samples(epoch+1)
                                
                # if (epoch + 1) % self.sampling_freq == 0:
                #     self._generate_samples(epoch+1)
                    
            if self.gpu_id == 0 and self.logs['val_loss'] < self.best_loss:
                print("replacing best checkpoint ...")
                self.best_loss = self.logs['val_loss']
                self._save_checkpoint(epoch+1,self.checkpoint_path)
                
                
                


def load_train_objs(args):
    if args.dataset == 'climate':
        train_set = E5(factor=args.factor, num_pred_steps=args.num_pred_steps,
                     scratch_dir='/pscratch/sd/v/vmikuni/FM/climate/train')
        val_set = E5(factor=args.factor, num_pred_steps=args.num_pred_steps,train=False,
                     scratch_dir='/pscratch/sd/v/vmikuni/FM/climate/valid')

    elif args.dataset == 'simple':
        train_set = Simple(factor=args.factor, num_pred_steps=args.num_pred_steps,
                           scratch_dir='/pscratch/sd/v/vmikuni/FM/simple')
        val_set = Simple(factor=args.factor, num_pred_steps=args.num_pred_steps,train=False,
                         scratch_dir='/pscratch/sd/v/vmikuni/FM/simple')

        
    else:
        train_set = NSKT(factor=args.factor, num_pred_steps=args.num_pred_steps,
                         scratch_dir=args.scratch_dir)
        val_set = NSKT(factor=args.factor, num_pred_steps=args.num_pred_steps,train=False,
                       scratch_dir=args.scratch_dir)
    unet_model,lowres_head,future_head, decoder_head = UNet(image_size=256, in_channels=1, out_channels=1, 
                                                            base_width=args.base_width,
                                                            num_pred_steps=args.num_pred_steps,
                                                            Reynolds_number=True)
        
    
    model = GaussianDiffusionModel(encoder_model=unet_model.cuda(),
                                   decoder_model = decoder_head.cuda(),
                                   lowres_model = lowres_head.cuda(),
                                   forecast_model = future_head.cuda(),
                                   betas=(1e-4, 0.02),
                                   n_T=args.time_steps, 
                                   prediction_type = args.prediction_type, 
                                   sampler = args.sampler,
                                   sample_loss = args.sample_loss,
                                   clip_loss = args.clip_loss)
    
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    factor = 1
    if args.fine_tune:
        factor = 4.
    optimizer = Lion(model.parameters(), lr=args.learning_rate/factor,betas=(0.95,0.98),weight_decay=0.01)
    return train_set,val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        sampler=DistributedSampler(dataset),
        num_workers=16
    )


def main(rank: int, world_size: int, sampling_freq: int, epochs: int, batch_size: int, run, args):


    local_rank, rank = ddp_setup(rank, world_size)
    device = torch.cuda.current_device()
    train_data,val_data,  model, optimizer = load_train_objs(args=args)
    model = model.to(device)

    if rank == 0:
        #==============================================================================
        # Model summary
        #==============================================================================
        print('**** Setup ****')
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')

    
    train_data = prepare_dataloader(train_data, batch_size)
    val_data = prepare_dataloader(val_data, batch_size)
    
    trainer = Trainer(model, train_data,val_data, optimizer,
                      rank,local_rank, sampling_freq, run, epochs = epochs,
                      run_name=args.run_name,scratch_dir = args.scratch_dir,fine_tune=args.fine_tune,dataset=args.dataset)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument("--run-name", type=str, default='run1', help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='nskt', help="Name of the dataset to train. Options are [nskt,climate,simple]")
    parser.add_argument("--scratch-dir", type=str, default='/pscratch/sd/v/vmikuni/FM/nskt_tensor/', help="Name of the current run.")
    parser.add_argument('--epochs', default=200, type=int, help='Total epochs to train the model')
    parser.add_argument('--sampling-freq', default=30, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch-size', default=16, type=int, help='Input batch size on each device (default: 32)')

    parser.add_argument('--num-pred-steps', default=3, type=int, help='different prediction steps to condition on')

    parser.add_argument('--factor', default=8, type=int, help='upsampling factor')
    parser.add_argument('--learning-rate', default=5e-5, type=int, help='learning rate')

    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")    
    parser.add_argument("--time-steps", type=int, default=2, help="Time steps for sampling")    
    parser.add_argument("--multi-node", action='store_true', default=False, help='Use multi node training')
    parser.add_argument("--fine_tune", action='store_true', default=False, help='Fine tune using pretrained model')
    parser.add_argument("--sample_loss", action='store_true', default=False, help='Run the model calling the generation step during training')
    parser.add_argument("--clip_loss", action='store_true', default=False, help='Run the model calling the generation step during training')
    
    parser.add_argument("--base-width", type=int, default=128, help="Basewidth of U-Net")    


    args = parser.parse_args()

    
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
            project="MoE_final",
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
        #Launch processes.
        print('Launching processes...')
        main(0,1,args.sampling_freq, args.epochs, args.batch_size, run, args)

