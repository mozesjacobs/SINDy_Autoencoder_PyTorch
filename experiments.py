import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import matplotlib.animation as animation

from cmd_line import parse_args
from src.dataset.Datasets import LabeledMNISTDataset, UCFDataset, SpritesDataset
from src.trainer.baseline import test
from src.utils.plotting import *
from src.utils.other import *
from src.utils.all_experiments import *


def main():
    # get args
    args = parse_args()

    # experiments output folder and checkpoint folder
    exp_folder = get_experiments_path(args)
    os.system("mkdir -p " + exp_folder)    
    if args.print_folder:
        print(exp_folder)
    cp_path, cp_folder = get_checkpoint_path(args)
    
    # load data
    _, test_set = load_data(args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
        
    # device
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()
    
    # load model
    net = make_model(args, device).to(device)
    checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))        
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    net.update_device(device)
    initial_e = checkpoint['epoch']

        
    if args.sample_grid == 1:
        # do the sample grid test
        sample_grid_test(net, test_loader, device, args, T=args.timesteps,
                         num=args.num_sample_grid,
                         file_name=exp_folder + "uncond_sample_grid.png",
                         gif_path=exp_folder + "sample_gif",
                         title="Unconditioned Samples")
        
    if args.sample_context_grid == 1:
        # do the sample context grid test
        sample_context_grid_test(net, test_loader, device, args, T=args.timesteps,
                                 num=args.num_sample_context_grid,
                                 file_name=exp_folder + "context_sample_grid.png",
                                 gif_path=exp_folder + "context_sample_gif",
                                 title="Unconditioned Samples")
        
    if args.inference_grid == 1:
        # do the inference grid test
        inference_grid_test(net, test_loader, device, args, T=args.timesteps,
                            num=args.num_inference_grid,
                            file_name=exp_folder + "inference_grid.png",
                            gif_path=exp_folder + "inference_gif",
                            title="Inference - Ground Truth, Prior, Posterior")
        

if __name__ == "__main__":
    main()
