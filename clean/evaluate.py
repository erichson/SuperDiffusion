import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from unet import UNet
from diffusion_model import GaussianDiffusionModel
from get_data import NSTK
from plotting import plot_samples

from train import load_train_objs
from torch_ema import ExponentialMovingAverage

def generate_samples(model,ema,dataset,args):
    with ema.average_parameters():
        model.eval()
        with torch.no_grad():
            PATH = "./train_samples_" + args.run_name
            if not os.path.exists(PATH):
                os.makedirs(PATH)
                        
            lowres_snapshots, snapshots, future_snapshots, s, Reynolds_number = next(iter(dataset))

            s = s.to('cuda')
            Reynolds_number = Reynolds_number.to('cuda')

            if args.superres:
                print("Performing super resolution")
                cond = lowres_snapshots.to('cuda')
                targets = snapshots.to('cuda')
            else:
                print("Performing forecasting")
                cond = snapshots.to('cuda')
                targets = future_snapshots.to('cuda')

            samples = model.sample(cond.shape[0], 
                                   (1, targets.shape[2], targets.shape[3]),
                                   cond, s, Reynolds_number,'cuda',superres=args.superres)
                        
    plot_samples(samples, cond, targets, PATH, 0)
    
 
def main(args):
    _,dataset, model, optimizer = load_train_objs( args=args)
    dataset = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=16
    )

    PATH = os.path.join(args.scratch_dir,'checkpoints',"checkpoint_" + args.run_name + ".pt")
    checkpoint = torch.load(PATH)
    print(checkpoint.keys())
    #optimizer.load_state_dict(checkpoint["optimizer"])
    model.base_model.load_state_dict(checkpoint["basemodel"])
    model.lowres_model.load_state_dict(checkpoint["lowres_model"])
    model.forecast_model.load_state_dict(checkpoint["forecast_model"])
    
    ema = ExponentialMovingAverage(model.parameters(),decay=0.999)
    ema.load_state_dict(checkpoint["ema"])
    generate_samples(model,ema,dataset,args)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument("--run-name", type=str, default='run1', help="Name of the current run.")
    parser.add_argument('--epochs', default=500, type=int, help='Total epochs to train the model')
    parser.add_argument('--sampling-freq', default=50, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch-size', default=16, type=int, help='Input batch size on each device (default: 32)')

    parser.add_argument('--superres', action='store_true', default=False, help='Superresolution')
    parser.add_argument('--factor', default=4, type=int, help='upsampling factor')
    parser.add_argument('--learning-rate', default=2e-4, type=int, help='learning rate')
    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")    
    parser.add_argument("--time-steps", type=int, default=1000, help="Time steps for sampling")
    parser.add_argument('--num-pred-steps', default=3, type=int, help='different prediction steps to condition on')
    parser.add_argument("--base-width", type=int, default=128, help="Basewidth of U-Net")
    parser.add_argument("--scratch-dir", type=str, default='/pscratch/sd/v/vmikuni/FM/nskt_tensor/', help="Name of the current run.")

    args = parser.parse_args()

    main(args)
