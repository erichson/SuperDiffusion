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
    def __init__(self, path, factor, train=True):
        super(NSTK_SR, self).__init__()
        self.path = path
        self.factor = factor
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
    
        if index < 1000:
            hr_img = torch.from_numpy(self.dataset[index, :, 0:1024, 0:1024]).float() # 512 x 512
        elif index < 2000:
            hr_img = torch.from_numpy(self.dataset[index-1000, :, 0:1024, 1024:2048]).float() # 512 x 512
        elif index < 3000:
            hr_img = torch.from_numpy(self.dataset[index-2000, :, 1024:2048, 0:1024]).float() # 512 x 512
        elif index < 4000:
            hr_img = torch.from_numpy(self.dataset[index-3000, :, 1024:2048, 1024:2048]).float() # 512 x 512            
            
        lr_img = hr_img[:, ::self.factor, ::self.factor]
        return lr_img, hr_img

    def __len__(self):
        return  1999 #self.length   