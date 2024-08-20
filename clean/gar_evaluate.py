#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:10:38 2024

@author: ben
"""


import os, sys
import torch


import numpy as np

from unet import UNet
from diffusion_model import GaussianDiffusionModel

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from plotting import plot_samples
import h5py
import scipy.stats
import matplotlib.pyplot as plt
import cmocean

from PIL import Image
from PIL import ImageDraw,ImageFont


def make_gif(PATH,args):
    targets = []
    forecasts = []
    for i in range(args.horizon-1):
        targets.append(Image.open(PATH + "/ddpm_forecast_target_"+ str(i) + ".png"))
        os.remove(PATH + "/ddpm_forecast_target_"+ str(i) + ".png")
        forecasts.append(Image.open(PATH + "/ddpm_forecast_pred_"+ str(i) + ".png"))
        os.remove(PATH + "/ddpm_forecast_pred_"+ str(i) + ".png")

    targets[0].save(PATH + f"/target_{args.step}_{args.horizon}.gif",
                    format='GIF',
                    append_images=targets[1:],
                    save_all=True,
                    duration=500, loop=0)

    forecasts[0].save(PATH + f"/forecast_{args.step}_{args.horizon}.gif",
                      format='GIF',
                      append_images=forecasts[1:],
                      save_all=True,
                      duration=500, loop=0)
    
        
        

class NSTK_eval(torch.utils.data.Dataset):
    def __init__(self, factor, 
                 num_pred_steps=3, 
                 patch_size=256, 
                 stride = 256,
                 step = 0,
                 horizon = 30,
                 Reynolds_number = 16000,
                 scratch_dir='./'
                 ):
        super(NSTK_eval, self).__init__()

        self.files_dict = {
            1000:os.path.join(scratch_dir,'1000_2048_2048_seed_3407.h5'),
            8000:os.path.join(scratch_dir, '8000_2048_2048_seed_3407.h5'),
            12000:os.path.join(scratch_dir,'12000_2048_2048_seed_3407.h5'),       
            16000:os.path.join(scratch_dir,'16000_2048_2048_seed_3407.h5'),       
            32000:os.path.join(scratch_dir,'32000_2048_2048_seed_987.h5'),        
        }

        assert Reynolds_number in self.files_dict, "ERROR: Reynolds number not present in evaluation datasets"
        self.file = self.files_dict[Reynolds_number]
        #print(self.file)
        
        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.patch_size = patch_size
        self.stride = stride
        self.step = step
        self.Reynolds_number = Reynolds_number
        self.horizon = horizon

        
        with h5py.File(self.file, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)
    
        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1    

        self.num_patches_per_image = ((self.data_shape[1] - self.patch_size) // self.stride + 1) * \
                                     ((self.data_shape[2] - self.patch_size) // self.stride + 1)

    
    def open_hdf5(self):
        self.dataset = h5py.File(self.file, 'r')['w']

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
        
        shift = self.step
        # Select a time index 
        #index = index // 100  
         
        # deterministic
        num_patches_per_row = (self.data_shape[2] - self.patch_size) // self.stride + 1
        image_idx = index // self.num_patches_per_image
        patch_idx = index % self.num_patches_per_image
        index = image_idx
        
        patch_row = (patch_idx // num_patches_per_row) * self.stride
        patch_col = (patch_idx % num_patches_per_row) * self.stride   
            
        patch = torch.from_numpy(self.dataset[index, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
        lowres_patch = patch[:, ::self.factor, ::self.factor]
        forecast = []
        for i in range(1,self.horizon):
            forecast.append(torch.from_numpy(self.dataset[index + shift*i, patch_row:(patch_row + self.patch_size),
                                                     patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0))
        

        return lowres_patch, patch, np.stack(forecast,1), F.one_hot(torch.tensor(shift),self.num_pred_steps), torch.tensor(self.Reynolds_number/40_000.)

    def __len__(self):
        return  self.num_patches_per_image  * 600 #30000 #self.length


def generate_samples(model,ema,dataset,args):
    PATH = "./train_samples_" + args.run_name
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    RFNE_error = []
    R2s = []
    i=0
    with ema.average_parameters():
        model.eval()
        with torch.no_grad():                
            for lowres_snapshots, snapshots, future_snapshots, s, Reynolds_number in tqdm(dataset):

                s = s.to('cuda')
                Reynolds_number = Reynolds_number.to('cuda')

                if args.superres:
                    cond = lowres_snapshots.to('cuda')
                    targets = snapshots.to('cuda')
                else:
                    cond = snapshots.to('cuda')
                    targets = future_snapshots.to('cuda')


                if args.superres:
                    predictions = model.sample(cond.shape[0], 
                                               (1, targets.shape[2], targets.shape[3]),
                                               cond, s, Reynolds_number,'cuda',superres=args.superres)
                    for j in range(predictions.shape[0]):
                        target = targets[j,0].cpu().detach().numpy()
                        prediction = predictions[j,0].cpu().numpy()                        
                        error = np.linalg.norm(prediction - target) / np.linalg.norm(target)
                        RFNE_error.append(error)

                    if i ==0:
                        PATH = "./train_samples_" + args.run_name
                        plot_samples(predictions, cond, targets, PATH, 0)

                else:
                    forecast = []                    
                    for step in range(args.horizon-1):
                        predictions = model.sample(cond.shape[0], 
                                                   (1, cond.shape[2], cond.shape[3]),
                                                   cond, s, Reynolds_number,'cuda',superres=args.superres)
                        forecast.append(predictions.cpu().numpy())
                        cond = predictions
                        
                        for j in range(predictions.shape[0]):
                            target = targets[j,0,step].cpu().detach().numpy()
                            prediction = predictions[j,0].cpu().numpy()
                            error = np.linalg.norm(prediction - target) / np.linalg.norm(target)
                            cc = scipy.stats.pearsonr(prediction.flatten(), target.flatten())[0]
                            RFNE_error.append(error)
                            R2s.append(cc)
                            #print(cc)
                            
                    if i ==0:                        
                        for im, img in enumerate(forecast):
                            fig = plt.figure(figsize=(12, 8))
                            plt.imshow(img[0,0,:,:], cmap=cmocean.cm.balance)
                            plt.tight_layout()
                            plt.savefig(PATH + "/ddpm_forecast_pred_"+ str(im) + ".png")
                            plt.close()
                            fig = plt.figure(figsize=(12, 8))
                            plt.imshow(targets[0,0,im].cpu().detach().numpy(), cmap=cmocean.cm.balance)
                            plt.tight_layout()
                            plt.savefig(PATH + "/ddpm_forecast_target_"+ str(im) + ".png")
                            plt.close()
                                                
                i+=1
                if i>=1:
                    break

        avg_RFNE = np.mean(RFNE_error)                    
        print(f'Average RFNE={avg_RFNE}')
        if args.superres ==False:
            avg_R2 =  np.mean(R2s, axis=0 )                  
            print(f'Average Pearson correlation coefficients={avg_R2}')
            make_gif(PATH,args)
             


    
    
def main(args, **kwargs):
    dataset = NSTK_eval(factor=args.factor,
                        num_pred_steps=args.num_pred_steps,
                        step = args.step,
                        Reynolds_number = args.Reynolds_number,
                        horizon=args.horizon,
                        scratch_dir=args.scratch_dir)

    unet_model,lowres_head,future_head = UNet(image_size=256, in_channels=1, out_channels=1, 
                                              base_width=args.base_width,
                                              num_pred_steps=args.num_pred_steps,
                                              Reynolds_number=True)

    model = GaussianDiffusionModel(base_model=unet_model.cuda(),
                                   lowres_model = lowres_head.cuda(),
                                   forecast_model = future_head.cuda(),
                                   betas=(1e-4, 0.02),
                                   n_T=args.time_steps, 
                                   prediction_type = args.prediction_type, 
                                   sampler = args.sampler)
    
    dataset = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=16
    )
    if 'checkpoint_path' in kwargs:
        PATH = kwargs['checkpoint_path']
    else:
        PATH = os.path.join(args.scratch_dir,'checkpoints',"checkpoint_" + args.run_name + ".pt")
    checkpoint = torch.load(PATH)
    #optimizer.load_state_dict(checkpoint["optimizer"])
    model.base_model.load_state_dict(checkpoint["basemodel"])
    model.lowres_model.load_state_dict(checkpoint["lowres_model"])
    model.forecast_model.load_state_dict(checkpoint["forecast_model"])
    
    ema = ExponentialMovingAverage(model.parameters(),decay=0.999)
    ema.load_state_dict(checkpoint["ema"])

    # set seed
    seed = 0
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    generate_samples(model,ema,dataset,args)


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument('--batch-size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--horizon', default=30, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument("--run-name", type=str, default='run1', help="Name of the current run.")
    parser.add_argument("--scratch-dir", type=str, default='../../', help="Name of the current run.")
    parser.add_argument('--superres', action='store_true', default=False, help='Superresolution')
    parser.add_argument('--num-pred-steps', default=3, type=int, help='different prediction steps to condition on')

    parser.add_argument('--Reynolds-number', default=16000, type=int, help='Reynolds number')
    parser.add_argument('--factor', default=4, type=int, help='upsampling factor')
    parser.add_argument('--step', default=1, type=int, help='future time steps to predict')

    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")    
    parser.add_argument("--time-steps", type=int, default=10, help="Time steps for sampling")    

    parser.add_argument("--base-width", type=int, default=64, help="Basewidth of U-Net")    

    args = parser.parse_args()
        
    checkpoint_path = '/data/rdl/NSTK/checkpoint_ddim_v_multinode_64.pt'
    main(args, checkpoint_path=checkpoint_path)       
   

