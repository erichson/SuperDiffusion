#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:11:28 2024

@author: ben
"""

import os
import numpy as np
import torch
import h5py

        
class NSTK(torch.utils.data.Dataset):
    def __init__(self, path):
        super(NSTK, self).__init__()
        self.path = path
        with h5py.File(self.path, 'r') as f:
            self.length = len(f['w'])        
        
    def open_hdf5(self):
        h5_file = h5py.File(self.path, 'r')
        self.dataset = h5_file['w']

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
        return torch.from_numpy(self.dataset[index, :, ::4, ::4]).float() # 512 x 512

    def __len__(self):
        return self.length    
    
    
    
class NSTK_SR(torch.utils.data.Dataset):
    def __init__(self, path, factor, train=True, patch_size = 256, stride = 256):
        super(NSTK_SR, self).__init__()
        self.path = path
        self.factor = factor
        self.train = train
        self.patch_size = patch_size
        self.stride = stride
        with h5py.File(self.path, 'r') as f:
            self.data_shape = f['w'].shape

            
        self.num_patches_per_image = ((self.data_shape[2] - self.patch_size) // self.stride + 1) * \
                                     ((self.data_shape[3] - self.patch_size) // self.stride + 1)

    def open_hdf5(self):
        h5_file = h5py.File(self.path, 'r')
        self.dataset = h5_file['w']

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
            
        if self.train:    
            index = index * 2
        else:
            index = index * 2 + 1


        num_patches_per_row = (self.data_shape[2] - self.patch_size) // self.stride + 1
        image_idx = index // self.num_patches_per_image
        patch_idx = index % self.num_patches_per_image
        
        patch_row = (patch_idx // num_patches_per_row) * self.stride
        patch_col = (patch_idx % num_patches_per_row) * self.stride
        
        hr_img = torch.from_numpy(self.dataset[image_idx, :, patch_row:patch_row + self.patch_size, patch_col:patch_col + self.patch_size]).float()
        lr_img = hr_img[:, ::self.factor, ::self.factor]
                
        return lr_img, hr_img

    def __len__(self):
        return  4000 #self.length   
    
    
    
    
class NSTK_Cast(torch.utils.data.Dataset):
    def __init__(self, factor, num_pred_steps=1, patch_size=256, stride = 128, train=True):
        super(NSTK_Cast, self).__init__()

        self.root = '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/'
        self.paths = ['2000_2048_2048_seed_2150.h5',
                      '4000_2048_2048_seed_2150.h5', 
                      '8000_2048_2048_seed_2150.h5', 
                      '16000_2048_2048_seed_2150.h5',
                      '32000_2048_2048_seed_2150.h5',
                      ]
        self.paths = [os.path.join(self.root, path) for path in self.paths]

        self.RN = [2000,4000,8000,16000,32000]
        
        
        
        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.train = train
        self.patch_size = patch_size
        self.stride = stride
        
        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)
    
        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1    

    
    def open_hdf5(self):
        self.datasets = [h5py.File(path, 'r')['w'] for path in self.paths]


    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
 
                         
        # Randomly select to super-resolve or forecast
        # superres = np.random.choice([True, False], size=1)[0]
        superres =  True
        
        if superres:
            shift = 0 #np.random.randint(0, self.num_pred_steps, 1)[0]
        else:
            # shift = np.random.randint(1, self.num_pred_steps, 1)[0]
            shift = 1
        
        # Select a time index 
        index = index // 75  
        
        if self.train:    
            index = index * 2
        else:
            index = index * 2 + 1            
            
            
        # Randomly select a patch from the image

        patch_row = np.random.randint(0, self.max_row) * self.stride
        patch_col = np.random.randint(0, self.max_col) * self.stride
        
        
        # Randomly select a dataset (Re=1000 or Re=16000)

        random_dataset = np.random.randint(0, len(self.paths))
        
        Reynolds_number = self.RN[random_dataset]
        dataset = self.datasets[random_dataset]
                                
        patch = torch.from_numpy(dataset[index, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
        target = torch.from_numpy(dataset[index + shift, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
            
        if superres:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            # return lr_patch, patch * 0, target, torch.tensor(shift), torch.tensor(Reynolds_number/40_000.)
            return lr_patch, target, torch.tensor(shift), torch.tensor(Reynolds_number/40_000.)
            
        else:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch * 0, patch, target, torch.tensor(shift), torch.tensor(Reynolds_number/40_000.)

    def __len__(self):
        return  55000 #30000 #self.length      
    
    
    

class SB_weather(torch.utils.data.Dataset):
    def __init__(self, factor, num_pred_steps=1, patch_size=256, stride=128, train=True):
        super(SB_weather, self).__init__()

        self.root = '/global/cfs/cdirs/m4633/foundationmodel/climate/train/'
        self.paths = [
            '2010.h5',
            '2008.h5',
            '2013.h5',
            '2011.h5'
        ]
        self.paths = [os.path.join(self.root, path) for path in self.paths]

        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.train = train
        self.patch_size = patch_size
        self.stride = stride

        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['fields'].shape
            print(self.data_shape)
    
        self.max_row = (self.data_shape[2] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[3] - self.patch_size) // self.stride + 1    

    def open_hdf5(self):
        self.datasets = [h5py.File(path, 'r')['fields'] for path in self.paths]

    def __getitem__(self, index):
        if not hasattr(self, 'datasets'):
            self.open_hdf5()
        
        superres = True
        
        if superres:
            shift = 0
        else:
            shift = 1
        
        index = index // 75

        if self.train:    
            index = index * 2
        else:
            index = index * 2 + 1

        # Randomly select a patch from the image
        patch_row = np.random.randint(0, self.max_row) * self.stride
        patch_col = np.random.randint(0, self.max_col) * self.stride
        
        # Randomly select a dataset (e.g., 2008, 2010, 2011, 2013)
        random_dataset = np.random.randint(0, len(self.paths))
        dataset = self.datasets[random_dataset]
        
        # Select a specific channel (e.g., channel 0, 1, or 2)
        channel = np.random.randint(0, 3)
        
        # Extract the patch and the target
        patch = torch.from_numpy(dataset[index, channel, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
        target = torch.from_numpy(dataset[index + shift, channel, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)

        if superres:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch, target, torch.tensor(shift), torch.tensor(channel)
        else:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch * 0, patch, target, torch.tensor(shift), torch.tensor(channel)

    def __len__(self):
        return 55000
    
class E5(torch.utils.data.Dataset):
    def __init__(self,
                 factor,
                 num_pred_steps=1,
                 patch_size=256,
                 stride = 128,
                 train=True,
                 scratch_dir='./'):
        super(E5, self).__init__()

        scratch_dir = '/global/cfs/cdirs/m4633/foundationmodel/climate/train/'
        self.paths = [os.path.join(scratch_dir,'2008.h5'),
                      os.path.join(scratch_dir,'2010.h5'),
                      os.path.join(scratch_dir,'2011.h5'),
                      os.path.join(scratch_dir,'2013.h5'),
                      ]
        self.RN = [0.0,0.0,0.0,0.0]
        
        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.train = train
        self.patch_size = patch_size
        self.stride = stride
        self.open_hdf5()
        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['fields'][:,1].shape

        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1


    def open_hdf5(self):
        self.datasets = [h5py.File(path, 'r')['fields'] for path in self.paths]

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()

        shift = np.random.randint(1, self.num_pred_steps + 1, 1)[0]

        # Select a time index                                                                                                                                                                                       
        index = index // 75

        # Randomly select a patch from the image                                                                                                                                                                    

        patch_row = np.random.randint(0, self.max_row) * self.stride
        patch_col = np.random.randint(0, self.max_col) * self.stride

        #Select one of the training files                                                                                                                                                                           
        random_dataset = np.random.randint(0, len(self.paths))

        Reynolds_number = self.RN[random_dataset]
        dataset = self.datasets[random_dataset]
        patch = torch.from_numpy(dataset[index, 1, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
        #patch = self.transform(patch)                                                                                                                                                                              
        future_patch = torch.from_numpy(dataset[index + shift,1, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)

        lowres_patch = patch[:,None, ::self.factor, ::self.factor]
        lowres_patch =  F.interpolate(lowres_patch,
                                      size=[patch.shape[1], patch.shape[2]],
                                      mode='bicubic')[:,0]

        return lowres_patch, patch, future_patch,  F.one_hot(torch.tensor(shift-1),self.num_pred_steps), torch.tensor(Reynolds_number/40000.)

    def __len__(self):
        return  27150