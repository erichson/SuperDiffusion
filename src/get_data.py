#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:11:28 2024

@author: ben
"""

import numpy as np
import torch
import h5py

    
class NSTK_train(torch.utils.data.Dataset):
    def __init__(self, factor, num_pred_steps=1, patch_size=256, stride = 128):
        super(NSTK_train, self).__init__()

        self.paths = ['../data/NSTK/2000_2048_2048_seed_2150.h5',
                      '../data/NSTK/4000_2048_2048_seed_2150.h5', 
                      '../data/NSTK/8000_2048_2048_seed_2150.h5', 
                      '../data/NSTK/16000_2048_2048_seed_2150.h5',
                      '../data/NSTK/32000_2048_2048_seed_2150.h5',
                      ]
        self.RN = [2000,4000,8000,16000,32000]
        self.factor = factor
        self.num_pred_steps = num_pred_steps
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
        superres = np.random.choice([True, False], size=1)[0]
        shift = 0 if superres else np.random.randint(1, self.num_pred_steps+1, 1)[0]
            
        # Select a time index 
        index = index // 75 * 2
            
        # Randomly select a patch from the image
        patch_row = np.random.randint(0, self.max_row) * self.stride
        patch_col = np.random.randint(0, self.max_col) * self.stride
        
        # Randomly select a dataset
        random_dataset = np.random.randint(0, len(self.paths))
        #random_dataset = np.random.choice(range(len(self.paths)), 1, p=np.array([.05, .15, .2, .3, .3]))[0]

        Reynolds_number = self.RN[random_dataset]
        dataset = self.datasets[random_dataset]
                                
        patch = torch.from_numpy(dataset[index, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
        target = torch.from_numpy(dataset[index + shift, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
            
        if superres:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch, patch * 0, target, torch.tensor(shift), torch.tensor(Reynolds_number/40_000.)
        else:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch * 0, patch, target, torch.tensor(shift), torch.tensor(Reynolds_number/40_000.)

    def __len__(self):
        return  55000      