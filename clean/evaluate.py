#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 13:10:38 2024

@author: ben
"""


import os, sys
import torch


import numpy as np

from unet import UNet
from diffusion_model import GaussianDiffusionModel

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from plotting import plot_samples
import h5py
import scipy.stats
import matplotlib.pyplot as plt
import cmocean

from PIL import Image
from PIL import ImageDraw,ImageFont
from get_data import E5_eval,NSKT_eval, Simple_eval


def make_gif(PATH,args):
    targets = []
    forecasts = []
    for i in range(args.horizon-1):
        targets.append(Image.open(PATH + "/ddpm_forecast_target_"+ str(i) + ".png"))
        os.remove(PATH + "/ddpm_forecast_target_"+ str(i) + ".png")
        forecasts.append(Image.open(PATH + "/ddpm_forecast_pred_"+ str(i) + ".png"))
        os.remove(PATH + "/ddpm_forecast_pred_"+ str(i) + ".png")

    targets[0].save(PATH + f"/target_{args.step}_{args.horizon}.gif",
                    format='GIF',
                    append_images=targets[1:],
                    save_all=True,
                    duration=500, loop=0)

    forecasts[0].save(PATH + f"/forecast_{args.step}_{args.horizon}.gif",
                      format='GIF',
                      append_images=forecasts[1:],
                      save_all=True,
                      duration=500, loop=0)
    
        
        


def generate_samples(model,ema,dataset,args):
    PATH = "./train_samples_" + args.run_name
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    RFNE_error = []
    R2s = []
    i=0
    with ema.average_parameters():
        model.eval()
        with torch.no_grad():                
            for lowres_snapshots, snapshots, future_snapshots, s, Reynolds_number in tqdm(dataset):

                s = s.to('cuda')
                Reynolds_number = Reynolds_number.to('cuda')

                if args.superres:
                    cond = lowres_snapshots.to('cuda')
                    targets = snapshots
                else:
                    cond = snapshots.to('cuda')
                    targets = future_snapshots


                if args.superres:
                    predictions = model.sample(cond.shape[0], 
                                               (1, targets.shape[2], targets.shape[3]),
                                               cond, s, Reynolds_number,'cuda',superres=args.superres)

                    for j in range(predictions.shape[0]):
                        target = targets[j,0].cpu().detach().numpy()
                        prediction = predictions[j,0].cpu().numpy()
                        error = np.linalg.norm(prediction - target) / np.linalg.norm(target)
                        RFNE_error.append(error)

                    if i ==0:
                        PATH = "./train_samples_" + args.run_name
                        plot_samples(predictions, cond, targets, PATH, 0)
                        samples = {
                            'conditioning_snapshots': cond.cpu().detach().numpy(),
                            'targets': targets.cpu().detach().numpy(),
                            'predictions': predictions.cpu().detach().numpy()
                        }

                        if not os.path.exists("./samples"):
                            os.makedirs("./samples")

                        np.save(
                            f'samples/samples_superres_RE_{args.Reynolds_number}_SR_{args.sampler}_{args.time_steps}_unet_{args.base_width}_' + str(i+1) + '.npy', samples)
                        print('saved samples')
                        

                else:
                    forecast = []

                    for _ in range(len(targets)):
                        predictions = model.sample(cond.shape[0], 
                                                   (1, cond.shape[2], cond.shape[3]),
                                                   cond, s, Reynolds_number,'cuda',
                                                   superres=args.superres)

                        forecast.append(predictions.cpu().numpy())
                        cond = predictions

                    for j in range(predictions.shape[0]):
                        RFNE_error_at_time_p = []
                        cc_error_at_time_p = []
                        for p in range(len(targets)):   
                            target = targets[p].cpu().detach().numpy()
                            prediction = forecast[p]
                            error = np.linalg.norm(prediction[j,0] - target[j,0]) / np.linalg.norm(target[j,0])
                            RFNE_error_at_time_p.append(error)
                            cc = scipy.stats.pearsonr(prediction[j,0].flatten(), target[j,0].flatten())[0]
                            cc_error_at_time_p.append(cc)
                            
                        RFNE_error.append(RFNE_error_at_time_p)   
                        R2s.append(cc_error_at_time_p)
                            
                    if i ==0:                        
                        for im, img in enumerate(forecast):
                            fig = plt.figure(figsize=(12, 8))
                            plt.imshow(img[0,0,:,:], cmap=cmocean.cm.balance)
                            plt.tight_layout()
                            plt.savefig(PATH + "/ddpm_forecast_pred_"+ str(im) + ".png")
                            plt.close()
                            fig = plt.figure(figsize=(12, 8))
                            plt.imshow(targets[im].cpu().detach().numpy()[0,0], cmap=cmocean.cm.balance)
                            plt.tight_layout()
                            plt.savefig(PATH + "/ddpm_forecast_target_"+ str(im) + ".png")
                            plt.close()
                                                
                i+=1
                # if i>=2:
                #     break

        if args.superres:
            avg_RFNE = np.mean(RFNE_error)                    
            print(f'Average RFNE={repr(avg_RFNE)}')
        else:
            avg_RFNE = np.mean(np.vstack(RFNE_error), axis=0 )
            print(f'Average RFNE={repr(avg_RFNE)}')
            avg_R2 =  np.mean(np.vstack(R2s), axis=0 )
            print(f'Average Pearson correlation coefficients={repr(avg_R2)}')
            make_gif(PATH,args)
             


    
    
def main(args):
    if args.dataset == 'climate':
        dataset = E5_eval(factor=args.factor,
                          num_pred_steps=args.num_pred_steps,
                          step = args.step,
                          Reynolds_number = args.Reynolds_number,
                          horizon=args.horizon,
                          scratch_dir="/pscratch/sd/v/vmikuni/FM/climate/test_1/",
                          superres=args.superres)

    elif args.dataset == 'simple':
        dataset = Simple_eval(factor=args.factor,
                              num_pred_steps=args.num_pred_steps,
                              step = args.step,
                              Reynolds_number = args.Reynolds_number,
                              horizon=args.horizon,
                              scratch_dir="/pscratch/sd/v/vmikuni/FM/simple",
                              superres=args.superres)


    else:
        dataset = NSKT_eval(factor=args.factor,
                            num_pred_steps=args.num_pred_steps,
                            step = args.step,
                            Reynolds_number = args.Reynolds_number,
                            horizon=args.horizon,
                            scratch_dir=args.scratch_dir,
                            superres=args.superres)

    unet_model,lowres_head,future_head, decoder = UNet(image_size=256, in_channels=1, out_channels=1, 
                                              base_width=args.base_width,
                                              num_pred_steps=args.num_pred_steps,
                                              Reynolds_number=True)

    model = GaussianDiffusionModel(encoder_model=unet_model.cuda(),
                                   lowres_model = lowres_head.cuda(),
                                   forecast_model = future_head.cuda(),
                                   decoder_model = decoder.cuda(),
                                   betas=(1e-4, 0.02),
                                   n_T=args.time_steps, 
                                   prediction_type = args.prediction_type, 
                                   sampler = args.sampler)
    
    dataset = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=16
    )

    PATH = os.path.join(args.scratch_dir,'checkpoints',f"checkpoint_{args.dataset}_{args.run_name}.pt")
    checkpoint = torch.load(PATH)
    #optimizer.load_state_dict(checkpoint["optimizer"])
    model.encoder_model.load_state_dict(checkpoint["basemodel"])
    model.lowres_model.load_state_dict(checkpoint["lowres_model"])
    model.forecast_model.load_state_dict(checkpoint["forecast_model"])
    model.decoder_model.load_state_dict(checkpoint["decoder_model"])
    
    ema = ExponentialMovingAverage(model.parameters(),decay=0.999)
    ema.load_state_dict(checkpoint["ema"])

    # set seed
    seed = 0
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    generate_samples(model,ema,dataset,args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument('--batch-size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--horizon', default=30, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument("--run-name", type=str, default='run1', help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='nskt', help="Name of the dataset for evaluation.")
    parser.add_argument("--scratch-dir", type=str, default='/pscratch/sd/v/vmikuni/FM/nskt_tensor/', help="Name of the current run.")
    parser.add_argument('--superres', action='store_true', default=False, help='Superresolution')
    parser.add_argument('--num-pred-steps', default=3, type=int, help='different prediction steps to condition on')

    parser.add_argument('--Reynolds-number', default=16000, type=int, help='Reynolds number')
    parser.add_argument('--factor', default=8, type=int, help='upsampling factor')
    parser.add_argument('--step', default=1, type=int, help='future time steps to predict')

    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")    
    parser.add_argument("--time-steps", type=int, default=2, help="Time steps for sampling")    

    parser.add_argument("--base-width", type=int, default=64, help="Basewidth of U-Net")    

    args = parser.parse_args()
        
    main(args)       
   

