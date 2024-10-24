#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:10:38 2024

@author: ben
"""


import os
import torch


import numpy as np

from src.unet import UNet
from src.diffusion_model import GaussianDiffusionModelCast
#from src.get_data import NSTK_SR as NSTK
from torch.utils.data import Dataset, DataLoader

from torch_ema import ExponentialMovingAverage
import scipy.stats

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

        self.paths = ['/data/rdl/NSTK/600_2048_2048_seed_3407.h5',
                      '/data/rdl/NSTK/1000_2048_2048_seed_3407.h5',
                      '/data/rdl/NSTK/2000_2048_2048_seed_3407.h5',
                      '/data/rdl/NSTK/4000_2048_2048_seed_3407.h5',
                      '/data/rdl/NSTK/8000_2048_2048_seed_3407.h5',
                      '/data/rdl/NSTK/12000_2048_2048_seed_3407.h5',
                      '/data/rdl/NSTK/16000_2048_2048_seed_3407.h5',
                      '/data/rdl/NSTK/24000_2048_2048_seed_3407.h5',
                      '/data/rdl/NSTK/32000_2048_2048_seed_3407.h5',
                      '/data/rdl/NSTK/36000_2048_2048_seed_3407.h5',
                      ]

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

        patch = torch.from_numpy(dataset[snapshot_idx, patch_row:(
            patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)


        if self.task == 'forecast':
            target = []
            for i in range(1, self.horizion):
                target.append(torch.from_numpy(dataset[snapshot_idx + (self.pred_steps*i), patch_row:(patch_row + self.patch_size),
                                                       patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0))
            return patch * 0, patch, target, torch.tensor(self.pred_steps), torch.tensor(Reynolds_number/40_000.)
        
        elif self.task == 'superres':
            lr_patch = patch[:, ::self.factor, ::self.factor]
            target = torch.from_numpy(dataset[snapshot_idx, patch_row:(
                patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
            return lr_patch, patch * 0, target, torch.tensor(self.pred_steps), torch.tensor(Reynolds_number/40_000.)

            

    def __len__(self):
        return self.num_patches_per_image * 70


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

    # Load Model
    unet_model = UNet(image_size=args.target_resolution, in_channels=1, out_channels=1,
                      base_width=args.base_width,
                      superres=True,
                      forecast=True,
                      num_pred_steps=args.num_pred_steps+1,
                      Reynolds_number=True)

    model = GaussianDiffusionModelCast(eps_model=unet_model.cuda(), betas=(1e-4, 0.02),
                                       n_T=args.time_steps,
                                       prediction_type=args.prediction_type,
                                       sampler=args.sampler)

    checkpoint = torch.load(
        f"checkpoints/checkpoint_run_FC_ddim10_unet{args.base_width}_v6.pt", weights_only=True)
    # optimizer.load_state_dict(checkpoint["optimizer"])
    model.eps_model.load_state_dict(checkpoint["model"])
    model.ema.load_state_dict(checkpoint["ema"])

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
        with model.ema.average_parameters():
            with torch.no_grad():
                model.eval()

                for i, (conditioning_snapshots, conditioning_snapshots2, targets, s, dat_class) in enumerate(testloader):

                    print(i)
                    conditioning_snapshots = conditioning_snapshots.to('cuda')
                    conditioning_snapshots2 = conditioning_snapshots2.to('cuda')
                    s = s.to('cuda')
                    dat_class = dat_class.to('cuda')

                    predictions = model.sample(conditioning_snapshots.shape[0],
                                               (1, args.target_resolution,
                                                args.target_resolution),
                                               conditioning_snapshots, 
                                               conditioning_snapshots2, 
                                               s, 
                                               dat_class,
                                               'cuda')

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

                        np.save(
                            f'samples/samples_superres_RE_{args.Reynolds_number}_SR_{args.sampler}_{args.time_steps}_unet_{args.base_width}_' + str(i+1) + '.npy', samples)
                        print('saved samples')

                    #if i == 10:
                    #    break

        avg_RFNE = np.mean(RFNE_error)
        print(f'Average RFNE={avg_RFNE}')

    elif args.task == 'forecast':

        # Get test data
        test_set = NSTK_FC(factor=args.factor,
                           task='forecast',
                           pred_steps=args.pred_steps,
                           horizion=args.horizen,
                           Reynolds_number=args.Reynolds_number,
                           stride=512,
                           )  # load your dataset

        testloader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=8,
        )

        RFNE_error = []
        R2s = []

        print(f'Number of batches: {len(testloader)}')
        # with model.module.ema.average_parameters():
        with model.ema.average_parameters():
            with torch.no_grad():
                model.eval()

                for i, (conditioning_snapshots, conditioning_snapshots2, targets, s, dat_class) in enumerate(testloader):

                    print(i)
                    conditioning_snapshots = conditioning_snapshots.to('cuda')
                    conditioning_snapshots2 = conditioning_snapshots2.to(
                        'cuda')
                    s = s.to('cuda')
                    dat_class = dat_class.to('cuda')

                    preds = []
                    for _ in range(len(targets)):
                        predictions = model.sample(conditioning_snapshots.shape[0],
                                                   (1, args.target_resolution,
                                                    args.target_resolution),
                                                   conditioning_snapshots, 
                                                   conditioning_snapshots2, 
                                                   s, 
                                                   dat_class,
                                                   'cuda')
                        preds.append(predictions.cpu().detach().numpy())
                        conditioning_snapshots2 = predictions

                    for j in range(predictions.shape[0]):
                        RFNE_error_at_time_p = []
                        cc_error_at_time_p = []
                        for p in range(len(targets)):

                            target = targets[p].cpu().detach().numpy()
                            prediction = preds[p]

                            # compute RFNE
                            error = np.linalg.norm(
                                prediction[j, 0, :, :] - target[j, 0, :, :]) / np.linalg.norm(target[j, 0, :, :])
                            RFNE_error_at_time_p.append(error)

                            # compute correlation coef
                            cc = scipy.stats.pearsonr(
                                prediction[j, 0, :, :].flatten(), target[j, 0, :, :].flatten())[0]
                            cc_error_at_time_p.append(cc)

                        RFNE_error.append(RFNE_error_at_time_p)
                        R2s.append(cc_error_at_time_p)

                    #print(np.mean(np.vstack(RFNE_error), axis=0 ))

                    if i == 0:
                        samples = {
                            'conditioning_snapshots': conditioning_snapshots.cpu().detach().numpy(),
                            'targets': targets,
                            'predictions': preds
                        }

                        if not os.path.exists("./samples"):
                            os.makedirs("./samples")

                        np.save(
                            f'samples/samples_forecast_RE_{args.Reynolds_number}_SR_{args.sampler}_{args.time_steps}_unet_{args.base_width}_' + str(i+1) + '.npy', samples)
                        print('saved samples')

                    #if i == 5:
                    #    break

        avg_RFNE = np.mean(np.vstack(RFNE_error), axis=0)
        print(f'Average RFNE={repr(avg_RFNE)}')

        avg_R2 = np.mean(np.vstack(R2s), axis=0)
        print(f'Average Pearson correlation coefficients={repr(avg_R2)}')


# export CUDA_VISIBLE_DEVICES=6; python evaluation.py --task forecast --batch-size 32 --horizen 25 --Reynolds-number 12000