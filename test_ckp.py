import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from src.unet import UNet
from src.diffusion_model import GaussianDiffusionModelSR
from src.get_data import NSTK_SR as NSTK
from src.plotting import plot_samples

from trainSR_nstk import load_train_objs


def generate_samples(model,dataset,args):
    with model.ema.average_parameters():
        model.eval()
        with torch.no_grad():
            PATH = "./train_samples_" + args.run_name
            if not os.path.exists(PATH):
                os.makedirs(PATH)
                        
            conditioning_snapshots, targets = next(iter(dataset))
            conditioning_snapshots = conditioning_snapshots.to('cuda')
        
            samples = model.sample(conditioning_snapshots.shape[0], 
                                   (1, targets.shape[2], targets.shape[3]), 
                                   conditioning_snapshots, 'cuda')
                        
    plot_samples(samples, conditioning_snapshots, targets, PATH, 0)
    
 
def main(args):
    dataset, model, optimizer = load_train_objs(superres=args.superres, args=args)
    dataset = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
    )
    PATH = "./checkpoints/" + "checkpoint_" + args.run_name + ".pt"
    checkpoint = torch.load(PATH)
    print(checkpoint.keys())
    optimizer.load_state_dict(checkpoint["optimizer"])
    model.eps_model.load_state_dict(checkpoint["model"])
    model.ema.load_state_dict(checkpoint["ema"])
    generate_samples(model,dataset,args)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument("--run-name", type=str, default='run1', help="Name of the current run.")
    parser.add_argument('--epochs', default=500, type=int, help='Total epochs to train the model')
    parser.add_argument('--sampling-freq', default=50, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch-size', default=16, type=int, help='Input batch size on each device (default: 32)')

    parser.add_argument('--superres', default=True, type=bool, help='Superresolution')
    parser.add_argument('--factor', default=4, type=int, help='upsampling factor')

    parser.add_argument('--learning-rate', default=2e-4, type=int, help='learning rate')

    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--sampler", type=str, default='ddim', help="Sampler to use to generate images")    
    parser.add_argument("--time-steps", type=int, default=1000, help="Time steps for sampling")    

    args = parser.parse_args()

    main(args)
