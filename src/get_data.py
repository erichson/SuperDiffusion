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
    def __init__(self, path, factor, train=True, patch_size = 256, stride = 256):
        super(NSTK_SR, self).__init__()
        self.path = path
        self.factor = factor
        self.train = train
        self.patch_size = patch_size
        self.stride = 1
        with h5py.File(self.path, 'r') as f:
            self.data_shape = f['w'].shape

            
        self.num_patches_per_image = ((self.data_shape[1] - self.patch_size) // self.stride + 1) * \
                                     ((self.data_shape[2] - self.patch_size) // self.stride + 1)

        
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
        self.path1 = '/data/rdl/NSTK/1000_2048_2048_seed_3407.h5'
        self.path2 = '/data/rdl/NSTK/8000_2048_2048_seed_3407.h5'
        self.path3 = '/data/rdl/NSTK/16000_2048_2048_seed_3407.h5'
        
        
        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.train = train
        self.patch_size = patch_size
        self.stride = stride
        
        with h5py.File(self.path1, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)
    
        with h5py.File(self.path2, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)

        with h5py.File(self.path3, 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)


        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1    

    
    def open_hdf5(self):
        h5_file1 = h5py.File(self.path1, 'r')
        self.dataset1 = h5_file1['w']

        h5_file2 = h5py.File(self.path2, 'r')
        self.dataset2 = h5_file2['w']        

        h5_file3 = h5py.File(self.path3, 'r')
        self.dataset3 = h5_file3['w']         

    def __del__(self):
        # Ensure the file is closed when the dataset object is deleted
        if self.dataset1:
            try:
                self.dataset1.close()
            except Exception as e:
                print(f"Failed to close HDF5 file 1: {e}")

        if self.dataset2:
            try:
                self.dataset2.close()
            except Exception as e:
                print(f"Failed to close HDF5 file 2: {e}")

        if self.dataset3:
            try:
                self.dataset3.close()
            except Exception as e:
                print(f"Failed to close HDF5 file 3: {e}")                
                

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
 
             
            
        # Randomly select to super-resolve or forecast
        superres = np.random.choice([True, False], size=1)[0]
        
        if superres:
            shift = 0 #np.random.randint(0, self.num_pred_steps, 1)[0]
        else:
            shift = np.random.randint(1, self.num_pred_steps, 1)[0]
            
        
        # Select a time index 
        index = index // 100  
        
        if self.train:    
            index = index * 4
        else:
            index = index * 2 + 1            
            
            
        # Randomly select a patch from the image

        patch_row = np.random.randint(0, self.max_row) * self.stride
        patch_col = np.random.randint(0, self.max_col) * self.stride
        
        
        # Randomly select a dataset (Re=1000 or Re=16000)
        dataset_choice = np.random.choice([0, 1, 2], size=1)[0]
        
        if dataset_choice == 0:
            dataset = self.dataset1
        elif dataset_choice == 1:
            dataset = self.dataset2
        else:
            dataset = self.dataset3
            
            
        patch = torch.from_numpy(dataset[index, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
        target = torch.from_numpy(dataset[index + shift, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
            

        if superres:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch, patch * 0, target, torch.tensor(shift), torch.tensor(dataset_choice)
        else:
            lr_patch = patch[:, ::self.factor, ::self.factor]
            return lr_patch * 0, patch, target, torch.tensor(shift), torch.tensor(dataset_choice)

    def __len__(self):
        return  30000 #30000 #self.length      
    
    
