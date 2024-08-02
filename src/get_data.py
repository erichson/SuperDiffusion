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
    def __init__(self, path, factor, num_pred_steps=1, train=True):
        super(NSTK_Cast, self).__init__()
        self.path = path
        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.train = train
        with h5py.File(self.path, 'r') as f:
            self.length = len(f['w'])        
        
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
    
        flag = False
        if index > 7200:
            index -= 7200
            flag = True
        
        shift = np.random.randint(0, self.num_pred_steps, 1)[0]
    
        #print(index)
        if index < 900:
            patch = torch.from_numpy(self.dataset[index, :, 0:256, 0:256]).float() # 256 x 256
            target = torch.from_numpy(self.dataset[index+shift, :, 0:256, 0:256]).float() # 256 x 256

        elif index < 1800:
            patch = torch.from_numpy(self.dataset[index-900, :, 0:256, 256:512]).float() # 256 x 256
            target = torch.from_numpy(self.dataset[index-900+shift, :, 0:256, 256:512]).float() # 256 x 256

        elif index < 2700:
            patch = torch.from_numpy(self.dataset[index-1800, :, 256:512, 0:256]).float() # 256 x 256
            target = torch.from_numpy(self.dataset[index-1800+shift, :, 256:512, 0:256]).float() # 256 x 256

        elif index < 3600:
            patch = torch.from_numpy(self.dataset[index-2700, :, 256:512, 256:512]).float() # 256 x 256 
            target = torch.from_numpy(self.dataset[index-2700+shift, :, 256:512, 256:512]).float() # 256 x 256            

        elif index < 4500:
            patch = torch.from_numpy(self.dataset[index-3600, :, 1792:2048, 1792:2048]).float() # 256 x 256 
            target = torch.from_numpy(self.dataset[index-3600+shift, :, 1792:2048, 1792:2048]).float() # 256 x 256            

        elif index < 5400:
            patch = torch.from_numpy(self.dataset[index-4500, :, 1792:2048, 256:512]).float() # 256 x 256     
            target = torch.from_numpy(self.dataset[index-4500+shift, :, 1792:2048, 256:512]).float() # 256 x 256            

        elif index < 6300:
            patch = torch.from_numpy(self.dataset[index-5400, :, 256:512, 1792:2048]).float() # 256 x 256      
            target = torch.from_numpy(self.dataset[index-5400+shift, :, 256:512, 1792:2048]).float() # 256 x 256            

        elif index <= 7200:
            patch = torch.from_numpy(self.dataset[index-6300, :, 512:768, 512:768]).float() # 256 x 256   
            target = torch.from_numpy(self.dataset[index-6300+shift, :, 512:768, 512:768]).float() # 256 x 256            

                                                             
        if flag:
            lr_patch = target[:, ::self.factor, ::self.factor]
            return lr_patch * 0, patch, target, torch.tensor(shift)
        else:
            lr_patch = target[:, ::self.factor, ::self.factor]
            return lr_patch, patch * 0, target, torch.tensor(shift)

    def __len__(self):
        return  7200 #self.length      
    
    
