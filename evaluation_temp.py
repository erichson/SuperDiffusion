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
# from src.diffusion_model import GaussianDiffusionModelCast
from src.afnonet import AFNONet
#from src.get_data import NSTK_SR as NSTK
from torch.utils.data import Dataset, DataLoader
import scipy.stats
from src.get_data import E5_eval

import h5py



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
    
    class Params:
        def __init__(self, patch_size, in_chans, out_chans):
            self.patch_size = patch_size
            self.N_in_channels = in_chans
            self.N_out_channels = out_chans
            self.num_blocks = 16

    params = Params(patch_size=256, in_chans=1, out_chans=1)
    model = AFNONet(params, img_size=(256, 256))
    
    trained_epoch = "30"
    checkpoint_path = f"/pscratch/sd/y/yanggao/SuperDiffusion/checkpoints/checkpoint_full1023interactive_{trained_epoch}.pt"
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
    
    if args.task == 'forecast':

        # Get test data
        # test_set = NSTK_FC(factor=args.factor,
        #                    task='forecast',
        #                    pred_steps=args.pred_steps,
        #                    horizion=args.horizen,
        #                    Reynolds_number=args.Reynolds_number,
        #                    stride=256,
        #                    )  # load your dataset
        
        test_set = E5_eval(factor=args.factor,
                           num_pred_steps=3)

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
        # with model.ema.average_parameters():
        with torch.no_grad():
            model.eval()

            for i, (_, conditioning_snapshot, targets, _, _) in enumerate(testloader):
                # conditioning_snapshot: torch.Size([32, 1, 256, 256])
                # targets: List(torch.Size([32, 1, 256, 256])), len=29

                # print(f"conditioning_snapshots.shape {conditioning_snapshot.shape}")
                # print(f"targets len {len(targets)}")
                # print(f"targets[0].shape {targets[0].shape}")

                print(f"Batch {i}")
                first_input=conditioning_snapshot
                conditioning_snapshot = conditioning_snapshot.to('cuda') 

                predictions = []
                for _ in range(len(targets)):
                    next_frame = model(conditioning_snapshot)
                    predictions.append(next_frame.cpu().detach().numpy())
                    conditioning_snapshot = next_frame # next_frame.shape torch.Size([32, 1, 256, 256])

                for j in range(next_frame.shape[0]):
                    RFNE_error_at_time_p = []
                    cc_error_at_time_p = []
                    for p in range(len(targets)):

                        target = targets[p].cpu().detach().numpy()
                        prediction = predictions[p]
                        
                        # print(f"target.shape {target.shape}")   
                        # print(f"prediction.shape {prediction.shape}")

                        # compute RFNE
                        error = np.linalg.norm(prediction[j, 0, :, :] - target[j, 0, :, :]) / np.linalg.norm(target[j, 0, :, :])
                        RFNE_error_at_time_p.append(error)

                        # compute correlation coef
                        cc = scipy.stats.pearsonr(prediction[j, 0, :, :].flatten(), target[j, 0, :, :].flatten())[0]
                        cc_error_at_time_p.append(cc)

                    RFNE_error.append(RFNE_error_at_time_p)
                    R2s.append(cc_error_at_time_p)

                print(np.mean(np.vstack(R2s), axis=0 ))

                if i == 0:
                    samples = {
                        'conditioning_snapshots': first_input.cpu().detach().numpy(),
                        'targets': targets,
                        'predictions': predictions
                    }
                    
                    # print(f"conditioning_snapshots2.shape {first_input.shape}")
                    # print(f"len targets {len(targets)}")    
                    # print(f"target.shape {targets[0].shape}")
                    # print(f"len preds {len(predictions)}")
                    # print(f"preds.shape {predictions[0].shape}")
                    
                    # conditioning_snapshots2.shape torch.Size([32, 1, 256, 256])
                    # len targets 29
                    # target.shape torch.Size([32, 1, 256, 256])
                    # len preds 29
                    # preds.shape (32, 1, 256, 256)

                    if not os.path.exists("./samples"):
                        os.makedirs("./samples")

                    file_name = f'samples/samples_fourcastnet_new' + trained_epoch + str(i+1) + '.npy'
                    np.save(file_name, samples)
                    print(f'saved samples at {file_name}')

                    #if i == 5:
                    #    break

        avg_RFNE = np.mean(np.vstack(RFNE_error), axis=0)
        print(f'Average RFNE={repr(avg_RFNE)}')

        avg_R2 = np.mean(np.vstack(R2s), axis=0)
        print(f'Average Pearson correlation coefficients={repr(avg_R2)}')


# export CUDA_VISIBLE_DEVICES=0; python evaluation_temp.py --task forecast --batch-size 32 --horizen 25 --Reynolds-number 0
