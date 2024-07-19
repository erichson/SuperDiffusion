#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:11:28 2024

@author: ben
"""

import torch
from torch.utils.data import DataLoader

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
        return torch.from_numpy(self.dataset[index, :, ::2, ::2]).float()

    def __len__(self):
        return self.length    