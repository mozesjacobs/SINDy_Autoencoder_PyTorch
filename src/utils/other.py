import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
#import sys
#sys.path.append('../')

#from src.dataset.Datasets import *

def load_data(args):
    # train and val data (using val as "test" data)
    if args.data_set == "lorenz":
        ds = read_data_sets("./data/smooth_moving_mnist/datasets.txt")
        data_paths = get_data_paths(args, ds)
        train_set = LabeledMNISTDataset(data_paths['train_data'], 
                 data_paths['train_labels'], args.labels)
        test_set = LabeledMNISTDataset(data_paths['val_data'],
                data_paths['val_labels'], args.labels)
    return train_set, test_set

def make_model(args, device):
    if args.model == 'SINDyAE':
        from src.models.VAE import KalmanGON

def get_lorenz_path():
    return "data/lorenz/train.npy", "data/lorenz/val.npy", "data/lorenz/test.npy"

def get_path_extra(args):
    labels = convert_labels_to_str(args.labels)
    return args.data_vers + "/" + labels
    #return labels

def get_checkpoint_path(args):
    cp_folder = args.model_folder + args.data_set + "/" + get_path_extra(args) + "/" + args.model + "/"
    cp_folder += args.decoder + "/" + args.session_name + "/"
    return cp_folder + 'checkpoint.pt', cp_folder

def get_args_path(args):
    args_folder = args.model_folder + args.data_set + "/" + get_path_extra(args) + "/" + args.model + "/"
    args_folder += args.decoder + "/" + args.session_name + "/"
    return args_folder + "args.txt", args_folder

def get_tb_path(args):
    train_name = args.tensorboard_folder + args.data_set + '/' + get_path_extra(args) + "/" + args.model + "/"
    train_name += args.decoder + "/" + args.session_name + '/train'
    test_name = args.tensorboard_folder + args.data_set + '/' + get_path_extra(args) + "/" + args.model + "/"
    test_name += args.decoder + "/" + args.session_name + '/val'
    return train_name, test_name

def get_experiments_path(args):
    exp_path = args.experiments + args.data_set + "/" + get_path_extra(args) + "/" + args.model + "/"
    return exp_path + args.decoder + "/" + args.session_name + "/"