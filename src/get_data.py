#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:11:28 2024

@author: ben
"""

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
    def __init__(self, path, factor, train=True,patch_size = 256,stride = 1):
        super(NSTK_SR, self).__init__()
        self.path = path
        self.factor = factor
        self.train = train
        self.patch_size = patch_size
        self.stride = 1
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


        num_patches_per_row = (self.data_shape[3] - self.patch_size) // self.stride + 1
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
    def __init__(self, path, factor, num_pred_steps=1, patch_size=256, train=True):
        super(NSTK_Cast, self).__init__()
        self.path = path
        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.train = train
        self.patch_size = patch_size
        with h5py.File(self.path, 'r') as f:
            self.length = len(f['w'])        
        
    def open_hdf5(self):
        h5_file = h5py.File(self.path, 'r')
        self.dataset = h5_file['w']

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
            
        index = index // 20            
            
        if self.train:    
            index = index * 2
        else:
            index = index * 2 + 1
    
    
        superres = np.random.choice([True,False], size=1)[0]
        
        if superres:
            shift = np.random.randint(0, self.num_pred_steps, 1)[0]
        else:
            shift = np.random.randint(1, self.num_pred_steps, 1)[0]
            
    
        # Randomly select a patch from the image
        max_row = 2048 - self.patch_size
        max_col = 2048 - self.patch_size
        patch_row = np.random.randint(0, max_row)
        patch_col = np.random.randint(0, max_col)
        
        patch = self.dataset[index, :, patch_row:patch_row + self.patch_size, patch_col:patch_col + self.patch_size]
        patch = torch.from_numpy(patch).float()

        target = self.dataset[index + shift, :, patch_row:patch_row + self.patch_size, patch_col:patch_col + self.patch_size]
        target = torch.from_numpy(target).float()


        if superres:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch, patch * 0, target, torch.tensor(shift)
        else:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch * 0, patch, target, torch.tensor(shift)

    def __len__(self):
        return  9000 #self.length      
    
    
