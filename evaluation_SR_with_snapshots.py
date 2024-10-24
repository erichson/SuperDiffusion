#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:10:38 2024

@author: ben
"""


import os
import torch


import numpy as np

import numpy as np

#from src.unet import UNet
#from src.diffusion_model import GaussianDiffusionModelCast
#from src.get_data import NSTK_SR as NSTK
from torch.utils.data import Dataset, DataLoader

# from torch_ema import ExponentialMovingAverage
import scipy.stats
from src.hat_arch import HAT

import h5py


class NSTK_FC(torch.utils.data.Dataset):
    def __init__(self, factor,
                 pred_steps=0,
                 patch_size=256,
                 stride=256,
                 task='superres',
                 horizion=30,
                 Reynolds_number=16000,
                 ):
        super(NSTK_FC, self).__init__()

        self.root = '/global/cfs/cdirs/m4633/foundationmodel/nskt_tensor/'
        self.paths = ['600_2048_2048_seed_3407.h5',
                      '1000_2048_2048_seed_3407.h5',
                      '2000_2048_2048_seed_3407.h5',
                      '4000_2048_2048_seed_3407.h5',
                      '8000_2048_2048_seed_3407.h5',
                      '12000_2048_2048_seed_3407.h5',
                      '16000_2048_2048_seed_3407.h5',
                      '24000_2048_2048_seed_3407.h5',
                      '32000_2048_2048_seed_3407.h5',
                      '36000_2048_2048_seed_3407.h5',
                      ]

        self.paths = [os.path.join(self.root, path) for path in self.paths]

        self.RN = [600, 1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 36000]

        self.factor = factor
        self.pred_steps = pred_steps
        self.patch_size = patch_size
        self.stride = stride
        self.task = task
        self.Reynolds_number = Reynolds_number
        self.horizion = horizion

        with h5py.File(self.paths[0], 'r') as f:
            self.data_shape = f['w'].shape
            print(self.data_shape)

        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1 

        self.num_patches_per_image = ((self.data_shape[1] - self.patch_size) // self.stride + 1) * \
                                     ((self.data_shape[2] - self.patch_size) // self.stride + 1)
                                     
        print(f'Number of patches per snapshot: {self.num_patches_per_image}')

    def open_hdf5(self):
        self.datasets = [h5py.File(path, 'r')['w'] for path in self.paths]


    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()

        # deterministic
        num_patches_per_row = (self.data_shape[2] - self.patch_size) // self.stride + 1
        if self.task == 'superres':
            snapshot_idx = (index // self.num_patches_per_image) * 8
        else:
            snapshot_idx = (index // self.num_patches_per_image) * 12 + 600             
        patch_idx = index % self.num_patches_per_image
        

        patch_row = (patch_idx // num_patches_per_row) * self.stride
        patch_col = (patch_idx % num_patches_per_row) * self.stride

        # Get Dataset
        dataset_idx = self.RN.index(self.Reynolds_number)
        Reynolds_number = self.RN[dataset_idx]
        dataset = self.datasets[dataset_idx]

        ###! CHANGE snapshot_idx here
        snapshot_idx = 250
        

        patch = torch.from_numpy(dataset[snapshot_idx, patch_row:(
            patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)



        lr_patch = patch[:, ::self.factor, ::self.factor]
        target = torch.from_numpy(dataset[snapshot_idx, patch_row:(
                patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
        return lr_patch, patch * 0, target, torch.tensor(self.pred_steps), torch.tensor(Reynolds_number/40_000.)

            

    def __len__(self):
        return self.num_patches_per_image * 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Input batch size on each device (default: 32)')

    parser.add_argument('--task', default='superres',
                        type=str, help='superres or forecast')
    parser.add_argument('--pred-steps', default=1, type=int,
                        help='prediction step for forecasting')
    parser.add_argument('--horizen', default=1, type=int,
                        help='orecasting horizen')

    parser.add_argument('--Reynolds-number', default=16000,
                        type=int, help='Reynolds number')
    parser.add_argument('--target-resolution', default=256,
                        type=int, help='target resolution')
    parser.add_argument('--factor', default=8, type=int,
                        help='upsampling factor')

    parser.add_argument("--prediction-type", type=str, default='v',
                        help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim',
                        help="Sampler to use to generate images")
    parser.add_argument("--time-steps", type=int, default=3,
                        help="Time steps for sampling")
    parser.add_argument('--num-pred-steps', default=3, type=int,
                        help='different prediction steps to condition on')

    parser.add_argument("--base-width", type=int,
                        default=64, help="Basewidth of U-Net")

    args = parser.parse_args()

    ### !!!
    checkpoint_path = "./checkpoints/checkpoint_full0930cont_60.pt"
    save_name = "samples_superres_HAT092550_"
    
    model = HAT(img_size=32, patch_size=1, in_chans=1,
            window_size=8, img_range=1., depths=[6, 6, 6, 6, 6, 6],
            embed_dim=256, num_heads=[8,8,8,8,8,8], mlp_ratio=4, upsampler='pixelshuffle',
            upscale=args.factor, resi_connection='1conv') #TODO

    print(f'Loading model from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # model.eps_model.load_state_dict(checkpoint["model"])
    # model.ema.load_state_dict(checkpoint["ema"])
    state_dict = checkpoint["model"]
    new_state_dict = {}
    for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)

    # set seed
    seed = 0
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    model.to('cuda')
    model.eval()

    if args.task == 'superres':

        # Get test data
        test_set = NSTK_FC(factor=args.factor,
                           task='superres',
                           pred_steps=0,
                           horizion=0,
                           Reynolds_number=args.Reynolds_number,
                           stride=256,
                           )  # load your dataset

        testloader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=8
        )

        RFNE_error = []
        print(f'Number of batches: {len(testloader)}')
        # with model.module.ema.average_parameters():
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
                    error = np.linalg.norm(predictions[j, 0, :, :].cpu().numpy(
                    ) - targets[j, 0, :, :].cpu().numpy()) / np.linalg.norm(targets[j, 0, :, :].cpu().numpy())
                    RFNE_error.append(error)
                print(np.mean(RFNE_error))

                if i == 0:
                    samples = {
                        'conditioning_snapshots': conditioning_snapshots.cpu().detach().numpy(),
                        'targets': targets.cpu().detach().numpy(),
                        'predictions': predictions.cpu().detach().numpy()
                    }

                    if not os.path.exists("./samples"):
                        os.makedirs("./samples")

                    file_name = f'samples/fullsnapshot_superres_RE_idx250_{args.Reynolds_number}_SR_{args.sampler}_{args.time_steps}_unet_{args.base_width}_' + str(i+1) + '.npy'
                    np.save(file_name, samples)
                    print(f'saved samples at {file_name}')

                    #if i == 10:
                    #    break

        avg_RFNE = np.mean(RFNE_error)
        print(f'Average RFNE={avg_RFNE}')



# export CUDA_VISIBLE_DEVICES=0; python evaluation_SR_with_snapshots.py --task superres --batch-size 64 --Reynolds-number 16000
