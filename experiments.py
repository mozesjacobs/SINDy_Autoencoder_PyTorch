import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from cmd_line import parse_args
from src.trainer.baseline import train, test
from src.utils.other import *#load_data, load_model, make_model, get_tb_path, get_checkpoint_path, get_args_path, get_experiments_path
from src.utils.model_utils import init_weights
from src.utils.experiment_utils import print_gov_eqs


def main():
    # get and save args
    args = parse_args()

    # train and val data (will refer to the val data as test data)
    _, test_set, _ = load_data(args)

    # dataloaders
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # device
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()

    # checkpoint, args, experiments path
    cp_path, cp_folder = get_checkpoint_path(args)
    args_path, args_folder = get_args_path(args)
    exp_folder = get_experiments_path(args)
    if not os.path.isdir(cp_folder):
        os.system("mkdir -p " + cp_folder)
    if not os.path.isdir(exp_folder):
        os.system("mkdir -p " + exp_folder)
    if args.print_folder == 1:
        print("Checkpoints saved at:        ", cp_folder)
        print("Experiment results saved at: ", exp_folder)

    # create model, optim, scheduler, initial epoch
    net = make_model(args).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.adam_regularization)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma_factor)
    initial_e = 0
    
    # load model, optim, scheduler, epoch from checkpoint
    if args.load_cp == 1:
        net, optim, scheduler, initial_e = load_model(net, optim, scheduler, cp_path, device)
    else:  # init network
        net.apply(init_weights)

    # print the governing equations
    print_gov_eqs(net)
    

if __name__ == "__main__":
    main()