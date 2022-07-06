import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import os
import json
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


from cmd_line import parse_args
from src.trainer.baseline import train, test
from src.utils.other import *
from src.utils.all_experiments import *

def main():
    # get and save args
    args = parse_args()

    # train and test data
    train_set, test_set = load_data(args)

    # dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # boards
    train_name, test_name = get_tb_path(args)
    train_board = SummaryWriter(train_name, purge_step=True)
    test_board = SummaryWriter(test_name, purge_step=True)

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
        print(cp_folder)
        print(exp_folder)

    # experiments output folder and checkpoint folder
    exp_folder = get_experiments_path(args)
    os.system("mkdir -p " + exp_folder)
    
    # save args
    with open(args_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # create model, optim, scheduler, initial epoch
    net = make_model(args, device).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.regularization)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma_factor)
    initial_e = 0
    
    # load model, optim, scheduler, epoch from checkpoint
    if args.load_cp == 1:
        checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))
        net.load_state_dict(checkpoint['model'])
        net.to(device)
        net.update_device(device)
        optim.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        #optim = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.regularization)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma_factor)
        initial_e = checkpoint['epoch']
    else:  # init network
        def init_weights(m):
            if isinstance(m, nn.Linear):# or isinstance(m, nn.ConvTranspose2d): 
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
                    #nn.init.xavier_uniform_(m.bias)
        net.apply(init_weights)

    #inference_grid_test(net, test_loader, device, args, T=args.timesteps,
    #                    num=args.num_inference_grid,
    #                    file_name=exp_folder + "inference_grid_idk.png",
    #                    gif_path=exp_folder + "inference_gif_idk",
    #                    title="Inference - Ground Truth, Prior, Posterior")

    # for each epoch
    for epoch in tqdm(range(args.epochs), desc="Epoch", total=args.epochs, dynamic_ncols=True):
        #net.update_kld_weight(args.weight_update, args.beta_max, args.specific_weight)
        #print("START")
        # train
        train(net, train_loader, train_board, optim, epoch + initial_e, args.clip, args.timesteps)
        #print("FINISHED EPOCH")
        #inference_grid_test(net, test_loader, device, args, T=args.timesteps,
        #                num=args.num_inference_grid,
        #                file_name=exp_folder + "inference_grid_aftertrain.png",
        #                gif_path=exp_folder + "inference_gif_aftertrain",
        #                title="Inference - Ground Truth, Prior, Posterior")
        # test
        if (epoch + 1) % args.test_interval == 0:
            test(net, test_loader, test_board, epoch + initial_e, args.timesteps)
        # inference grid
        if (epoch + 1) % args.inference_grid_interval == 0:
            inference_grid_test(net, test_loader, device, args, T=args.timesteps,
                                num=args.num_inference_grid,
                                file_name=exp_folder + "inference_grid_e" + str(epoch + initial_e) + ".png",
                                gif_path=exp_folder + "inference_gif_e" + str(epoch + initial_e),
                                title="Inference - Ground Truth, Prior, Posterior")
        # sample grid
        if (epoch + 1) % args.sample_grid_interval == 0:
            sample_grid_test(net, test_loader, device, args, T=args.timesteps,
                             num=args.num_sample_grid,
                             file_name=exp_folder + "uncond_sample_grid_e" + str(epoch + initial_e) + ".png",
                             gif_path=exp_folder + "sample_gif_e" + str(epoch + initial_e),
                             title="Unconditioned Samples")
        # sample context grid
        if (epoch + 1) % args.sample_context_grid_interval == 0:
            sample_context_grid_test(net, test_loader, device, args, T=args.timesteps,
                                     num=args.num_sample_context_grid,
                                     file_name=exp_folder + "context_sample_grid_e" + str(epoch + initial_e) + ".png",
                                     gif_path=exp_folder + "context_sample_gif_e" + str(epoch + initial_e),
                                     title="Unconditioned Samples")
        # step on learning rate scheduler
        scheduler.step()
        # update kld weight
        #print("UPDATE")
    
        # save checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0:
            checkpoint = {'epoch': epoch + initial_e,
                          'model': net.state_dict(),
                          'optimizer': optim.state_dict(),
                          'scheduler': scheduler.state_dict()}
            torch.save(checkpoint, cp_path)

        # update kld
        net.update_kld_weight(args.weight_update, args.beta_max, args.specific_weight)
        

if __name__ == "__main__":
    main()