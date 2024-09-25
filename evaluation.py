#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:10:38 2024

@author: ben
"""


import os
import torch


import numpy as np
# from src.unet import UNet
# from src.diffusion_model import GaussianDiffusionModelCast
#from src.get_data import NSTK_SR as NSTK
from torch.utils.data import Dataset, DataLoader
from src.swinIR import SwinIR
# from torch_ema import ExponentialMovingAverage
import scipy.stats

import h5py


class NSTK_FC(torch.utils.data.Dataset):
    def __init__(self, factor, 
                 num_pred_steps=1, 
                 patch_size=256, 
                 stride = 256,
                 mode = 'superres',
                 step = 0,
                 horizion = 7,
                 Reynolds_number = 16000,
                 train=False,
                 in_domain = False
                 ):
        super(NSTK_FC, self).__init__()
        
        if in_domain:
            self.path1 = '/data/rdl/NSTK/1000_2048_2048_seed_2150.h5'
            self.path2 = '/data/rdl/NSTK/8000_2048_2048_seed_2150.h5'
            self.path3 = '/data/rdl/NSTK/16000_2048_2048_seed_2150.h5'    
        else:
            self.path1 = '/data/rdl/NSTK/1000_2048_2048_seed_3407.h5'
            self.path2 = '/data/rdl/NSTK/8000_2048_2048_seed_3407.h5'
            self.path3 = '/data/rdl/NSTK/12000_2048_2048_seed_3407.h5'        
            self.path4 = '/data/rdl/NSTK/16000_2048_2048_seed_3407.h5'      
            self.path5 = '/data/rdl/NSTK/24000_2048_2048_seed_3407.h5'   
            self.path6 = '/data/rdl/NSTK/32000_2048_2048_seed_3407.h5'     

        
        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.train = train
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.step = step
        self.Reynolds_number = Reynolds_number
        self.horizion = horizion

        
        with h5py.File(self.path1, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)
    
        with h5py.File(self.path2, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)

        with h5py.File(self.path3, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)

        with h5py.File(self.path4, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)

        with h5py.File(self.path5, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)
            
        with h5py.File(self.path6, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)


        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1    

        self.num_patches_per_image = ((self.data_shape[1] - self.patch_size) // self.stride + 1) * \
                                     ((self.data_shape[2] - self.patch_size) // self.stride + 1)

    
    def open_hdf5(self):
        h5_file1 = h5py.File(self.path1, 'r')
        self.dataset1 = h5_file1['w']

        h5_file2 = h5py.File(self.path2, 'r')
        self.dataset2 = h5_file2['w']  

        h5_file3 = h5py.File(self.path3, 'r')
        self.dataset3 = h5_file3['w']  

        h5_file4 = h5py.File(self.path4, 'r')
        self.dataset4 = h5_file4['w']  

        h5_file5 = h5py.File(self.path5, 'r')
        self.dataset5 = h5_file5['w']  
        
        h5_file6 = h5py.File(self.path6, 'r')
        self.dataset6 = h5_file6['w']


    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
 
             
            
        
        shift = self.step
        
        
        # Select a time index 
        #index = index // 100  
        
        if self.train:    
            index = index * 4
        else:
            index = index * 2 + 1            
            
            
        # Randomly select a patch from the image
        #patch_row = np.random.randint(0, self.max_row) * self.stride
        #patch_col = np.random.randint(0, self.max_col) * self.stride
 
        # deterministic
        num_patches_per_row = (self.data_shape[2] - self.patch_size) // self.stride + 1
        image_idx = index // self.num_patches_per_image
        patch_idx = index % self.num_patches_per_image
        index = image_idx
        
        patch_row = (patch_idx // num_patches_per_row) * self.stride
        patch_col = (patch_idx % num_patches_per_row) * self.stride   
        
        
        if self.Reynolds_number == 1000:
            dataset = self.dataset1
            Reynolds_number = 1000
        elif self.Reynolds_number == 8000:
            dataset = self.dataset2
            Reynolds_number = 8000
        elif self.Reynolds_number == 12000:
            dataset = self.dataset3  
            Reynolds_number = 12000
        elif self.Reynolds_number == 16000:
            dataset = self.dataset4
            Reynolds_number = 16000
        elif self.Reynolds_number == 24000:
            dataset = self.dataset5  
            Reynolds_number = 24000
        elif self.Reynolds_number == 32000:
            dataset = self.dataset6  
            Reynolds_number = 32000
            
        patch = torch.from_numpy(dataset[index, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
        target = torch.from_numpy(dataset[index + shift, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)

        
        if self.mode == 'forecast':
            target = []
            for i in range(1,self.horizion):
                target.append(torch.from_numpy(dataset[index + (shift*i), patch_row:(patch_row + self.patch_size),
                                                       patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0))

            

        if self.mode == 'superres':
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch, patch * 0, target, torch.tensor(shift), torch.tensor(Reynolds_number/40_000.)
        else:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch * 0, patch, target, torch.tensor(shift), torch.tensor(Reynolds_number/40_000.)

    def __len__(self):
        return  self.num_patches_per_image  * 600 #30000 #self.length          
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument('--batch-size', default=128, type=int, help='Input batch size on each device (default: 32)')

    parser.add_argument('--mode', default='superres', type=str, help='superres or forecast')
    parser.add_argument('--pred-step', default=1, type=int, help='prediction step for forecasting')
    parser.add_argument('--horizen', default=1, type=int, help='orecasting horizen')


    parser.add_argument('--Reynolds-number', default=16000, type=int, help='Reynolds number')
    parser.add_argument('--target-resolution', default=256, type=int, help='target resolution')
    parser.add_argument('--factor', default=8, type=int, help='upsampling factor')


    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")    
    parser.add_argument("--time-steps", type=int, default=2, help="Time steps for sampling")    
    parser.add_argument('--num-pred-steps', default=1, type=int, help='different prediction steps to condition on')


    parser.add_argument("--base-width", type=int, default=64, help="Basewidth of U-Net")    

    args = parser.parse_args()

        
       
   


    # Load Model
    model = SwinIR(img_size=32, patch_size=1, in_chans=1,
                   window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                   embed_dim=256, num_heads=[8,8,8,8,8,8], mlp_ratio=4, upsampler='pixelshuffle',
                   upscale=8,resi_connection='1conv') 
    
    print('**** Setup ****')
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    print('************')
    
    
    checkpoint = torch.load(f"/data/home/dwlyu/SuperDiffusion/checkpoints/checkpoint_sr_1.pt", weights_only=True)
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)
    #optimizer.load_state_dict(checkpoint["optimizer"])
    # model.eps_model.load_state_dict(checkpoint["model"])
    # model.ema.load_state_dict(checkpoint["ema"])

    # set seed
    seed = 0
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    
    
    
    model.to('cuda')
    model.eval()
    
    if args.mode == 'superres':
        
        # Get test data
        test_set = NSTK_FC(factor=args.factor, 
                           train=False,
                           mode = 'superres',
                           step = 0,
                           horizion = 0,
                           Reynolds_number = args.Reynolds_number,
                           )  # load your dataset
        
        
        testloader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=8
        ) 
    
        RFNE_error = []        
        with torch.no_grad():
                model.eval()               
                
                for i, (conditioning_snapshots, conditioning_snapshots2, targets, s, dat_class) in enumerate(testloader):
                    
                        print(i)
                        conditioning_snapshots = conditioning_snapshots.to('cuda')
                        conditioning_snapshots2 = conditioning_snapshots2.to('cuda')
                        s = s.to('cuda')
                        dat_class = dat_class.to('cuda')
            
                
                        predictions = model(conditioning_snapshots)
                    
                        
                        for j in range(predictions.shape[0]):
                            error = np.linalg.norm(predictions[j,0,:,:].cpu().numpy() - targets[j,0,:,:].cpu().numpy()) / np.linalg.norm(targets[j,0,:,:].cpu().numpy())
                            RFNE_error.append(error)
                        print(np.mean(RFNE_error) )
                            
                    
                        if i == 0:
                            samples = {
                                'conditioning_snapshots': conditioning_snapshots.cpu().detach().numpy(),
                                'targets': targets.cpu().detach().numpy(),
                                'predictions': predictions.cpu().detach().numpy()
                                }
                        
                            if not os.path.exists("./samples"):
                                        os.makedirs("./samples")
                                        
                                        
                            np.save(f'samples/samples_superres_RE_{args.Reynolds_number}_SR_{args.sampler}_{args.time_steps}_unet_{args.base_width}_' + str(i+1) + '.npy', samples)
                            print('saved samples')
          
                        if i == 10:
                          break
          
        avg_RFNE = np.mean(RFNE_error)                    
        print(f'Average RFNE={avg_RFNE}')
        
        
    elif args.mode == 'forecast':
        
        # Get test data
        test_set = NSTK_FC(factor=args.factor, 
                           train=False,
                           mode = 'forecast',
                           step = args.pred_step,
                           horizion = args.horizen,
                           Reynolds_number = args.Reynolds_number,
                           )  # load your dataset
        
        
        testloader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=8
        ) 
    
        RFNE_error = []
        R2s = []
    
        #with model.module.ema.average_parameters():          
        with torch.no_grad():
                model.eval()               
                
                for i, (conditioning_snapshots, conditioning_snapshots2, targets, s, dat_class) in enumerate(testloader):
                    
                        print(i)
                        conditioning_snapshots = conditioning_snapshots.to('cuda')
                        conditioning_snapshots2 = conditioning_snapshots2.to('cuda')
                        s = s.to('cuda')
                        dat_class = dat_class.to('cuda')
            
                
                        #predictions = model.sample(conditioning_snapshots.shape[0], 
                        #                           (1, args.target_resolution, args.target_resolution), 
                        #                           conditioning_snapshots, conditioning_snapshots2, s, dat_class, 'cuda')
                        predictions = model(conditioning_snapshots, conditioning_snapshots2)
                    

                        preds = []
                        for _ in range(len(targets)):
                            predictions = model(conditioning_snapshots, conditioning_snapshots2)
                            preds.append(predictions.cpu().detach().numpy())
                            conditioning_snapshots2 = predictions
            
                
                        for j in range(predictions.shape[0]):
                            RFNE_error_at_time_p = []
                            cc_error_at_time_p = []
                            for p in range(len(targets)):
                                
                                target = targets[p].cpu().detach().numpy()
                                prediction = preds[p]
                                
                                # compute RFNE 
                                error = np.linalg.norm(prediction[j,0,:,:] - target[j,0,:,:]) / np.linalg.norm(target[j,0,:,:])
                                RFNE_error_at_time_p.append(error)
                                
                                # compute correlation coef
                                cc = scipy.stats.pearsonr(prediction[j,0,:,:].flatten(), target[j,0,:,:].flatten())[0]
                                cc_error_at_time_p.append(cc)

                                
                            RFNE_error.append(RFNE_error_at_time_p)   
                            R2s.append(cc_error_at_time_p)
                            
                        #print(np.mean(np.vstack(RFNE_error), axis=0 ))
                            
                    
                        if i == 0:
                            samples = {
                                'conditioning_snapshots': conditioning_snapshots.cpu().detach().numpy(),
                                'targets': targets,
                                'predictions': preds
                                }
                        
                            if not os.path.exists("./samples"):
                                        os.makedirs("./samples")
                                        
                                        
                            np.save(f'samples/samples_forecast_RE_swinIR' + '.npy', samples)
                            print('saved samples')
          
                        if i == 10:
                          break
          
        avg_RFNE =  np.mean(np.vstack(RFNE_error), axis=0 )                  
        print(f'Average RFNE={avg_RFNE}')   
        
        avg_R2 =  np.mean(np.vstack(R2s), axis=0 )                  
        print(f'Average Pearson correlation coefficients={avg_R2}')