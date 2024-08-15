#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:11:28 2024

@author: ben
"""

import numpy as np
import torch
import h5py
import os
import torch.nn.functional as F
            
class NSTK(torch.utils.data.Dataset):
    def __init__(self,
                 factor,
                 num_pred_steps=1,
                 patch_size=256,
                 stride = 128,
                 train=True,
                 scratch_dir='./'):
        super(NSTK, self).__init__()
        
        self.paths = [os.path.join(scratch_dir,'1000_2048_2048_seed_2150.h5'),
                      os.path.join(scratch_dir,'8000_2048_2048_seed_2150.h5'),
                      os.path.join(scratch_dir,'16000_2048_2048_seed_2150.h5'),
                      ]

        self.RN = [1000,8000,32000]
        
        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.train = train
        self.patch_size = patch_size
        self.stride = stride
        
        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['w'].shape

        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1    

    
    def open_hdf5(self):
        self.datasets = [h5py.File(path, 'r')['w'] for path in self.paths]

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
 
        shift = np.random.randint(1, self.num_pred_steps, 1)[0]
                        
        # Select a time index 
        index = index // 75  
        
        if self.train:    
            index = index * 2
        else:
            index = index * 2 + 1            
            
            
        # Randomly select a patch from the image

        patch_row = np.random.randint(0, self.max_row) * self.stride
        patch_col = np.random.randint(0, self.max_col) * self.stride
        
        #Select one of the training files
        random_dataset = np.random.randint(0, len(self.paths))
        
        Reynolds_number = self.RN[random_dataset]
        dataset = self.datasets[random_dataset]

            
        patch = torch.from_numpy(dataset[index, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0) 
        future_patch = torch.from_numpy(dataset[index + shift, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)            
        lowres_patch = patch[:, ::self.factor, ::self.factor]
        return lowres_patch, patch, future_patch,  F.one_hot(torch.tensor(shift),self.num_pred_steps), torch.tensor(Reynolds_number/40_000.)

    def __len__(self):
        return  45000 #30000 #self.length      
    
    
