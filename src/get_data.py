#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:11:28 2024

@author: ben
"""

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
    def __init__(self, path, factor):
        super(NSTK_SR, self).__init__()
        self.path = path
        self.factor = factor
        with h5py.File(self.path, 'r') as f:
            self.length = len(f['w'])        
        
    def open_hdf5(self):
        h5_file = h5py.File(self.path, 'r')
        self.dataset = h5_file['w']

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
        hr_img = torch.from_numpy(self.dataset[index, :, 0:1024:2, 0:1024:2]).float() # 512 x 512
        lr_img = hr_img[:, ::self.factor, ::self.factor]
        return lr_img, hr_img

    def __len__(self):
        return self.length        