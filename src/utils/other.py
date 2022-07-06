import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os
import sys
sys.path.append('../')

from src.dataset.Datasets import *

def load_data(args):
    # train and val data (using val as "test" data)
    if args.data_set == "smooth_moving_mnist":
        ds = read_data_sets("./data/smooth_moving_mnist/datasets.txt")
        data_paths = get_data_paths(args, ds)
        train_set = LabeledMNISTDataset(data_paths['train_data'], 
                 data_paths['train_labels'], args.labels)
        test_set = LabeledMNISTDataset(data_paths['val_data'],
                data_paths['val_labels'], args.labels)
    elif args.data_set == "ucf101":
        train_set = UCFDataset(data_dir='../Novelty-Eval/train_data_5/',
                               metadata_name="train_metadata_fixed.csv",
                               frame_size=(args.frame_size, args.frame_size),
                               num_classes=31,
                               transform=None,
                               chosen_ont_ids=args.labels)
        test_set = UCFDataset(data_dir='../Novelty-Eval/test_data_5/',
                              metadata_name="test_metadata.csv",
                              frame_size=(args.frame_size, args.frame_size),
                              num_classes=31,
                              transform=None,
                              chosen_ont_ids=args.labels)
    elif args.data_set == "sprites":
        train_set = SpritesDataset(data_dir='../Sprites/npy/',
                                   labels=args.labels,
                                   train=True)
        test_set = SpritesDataset(data_dir='../Sprites/npy/',
                                  labels=args.labels,
                                  train=False)
    return train_set, test_set

def make_model(args, device):
    if args.model == 'VAE':
        from src.models.VAE import KalmanGON
    if args.model == 'HKG':
        from src.models.HKG import KalmanGON
    if args.model == 'HyperKG0_14':
        from src.models.HyperKG0_14 import KalmanGON
    if args.model == 'HyperKG0_13':
        from src.models.HyperKG0_13 import KalmanGON
    if args.model == 'HyperKG0_12':
        from src.models.HyperKG0_12 import KalmanGON
    if args.model == 'HyperKG0_11':
        from src.models.HyperKG0_11 import KalmanGON
    if args.model == 'HyperKG0_10':
        from src.models.HyperKG0_10 import KalmanGON
    if args.model == 'HyperKG0_9':
        from src.models.HyperKG0_9 import KalmanGON
    if args.model == 'HyperKG0_8':
        from src.models.HyperKG0_8 import KalmanGON
    if args.model == 'HyperKG0_7':
        from src.models.HyperKG0_7 import KalmanGON
    if args.model == 'HyperKG0_6':
        from src.models.HyperKG0_6 import KalmanGON
    if args.model == 'HyperKG_5':
        from src.models.HyperKG_5 import KalmanGON
    if args.model == 'HyperKG0_5':
        from src.models.HyperKG0_5 import KalmanGON
    if args.model == 'HyperKG0_4':
        from src.models.HyperKG0_4 import KalmanGON
    if args.model == 'HyperKG0_3':
        from src.models.HyperKG0_3 import KalmanGON
    if args.model == 'HyperKG0_2':
        from src.models.HyperKG0_2 import KalmanGON
    if args.model == 'HyperKG0':
        from src.models.HyperKG0 import KalmanGON
    if args.model == 'KG0_object2':
        from src.models.KG0_object2 import KalmanGON
    if args.model == 'KG0_object':
        from src.models.KG0_object import KalmanGON
    if args.model == 'KG0_2':
        from src.models.KG0_2 import KalmanGON
    if args.model == 'KG02':
        from src.models.KG02 import KalmanGON
    if args.model == 'KG0':
        from src.models.KG0 import KalmanGON
    if args.model == 'ConvKG':
        from src.models.ConvKG import KalmanGON
    if args.model == 'ConvKG2':
        from src.models.ConvKG2 import KalmanGON
    if args.model == 'ConvKG3':
        from src.models.ConvKG3 import KalmanGON
    if args.model == 'ConvKG4':
        from src.models.ConvKG4 import KalmanGON
    if args.model == 'ConvKG5':
        from src.models.ConvKG5 import KalmanGON
    if args.model == 'ConvKG6':
        from src.models.ConvKG6 import KalmanGON
    if args.model == 'ConvKG7':
        from src.models.ConvKG7 import KalmanGON
    if args.model == 'ConvKG8':
        from src.models.ConvKG8 import KalmanGON
    if args.model == 'CKG0_1':
        from src.models.CKG0_1 import KalmanGON
    if args.model == 'CKG0_3':
        from src.models.CKG0_3 import KalmanGON
    if args.model == 'CKG0':
        from src.models.CKG0 import KalmanGON
    elif args.model == 'KG':
        from src.models.KG import KalmanGON
    return KalmanGON(args, device)


def load_decoder(args):
    if args.decoder == '18':
        from src.models.decoder import Decoder18
        return Decoder18(args)
    if args.decoder == '17':
        from src.models.decoder import Decoder17
        return Decoder17(args.r_dim, args.frame_size, args.hidden_dim, args.channels)
    if args.decoder == '16':
        from src.models.decoder import Decoder16
        return Decoder16(args.r_dim, args.frame_size, args.hidden_dim, args.channels)
    if args.decoder == '15':
        from src.models.decoder import Decoder15
        return Decoder15(args.latent_channels, args.decoder_channels, args.channels)
    if args.decoder == '14':
        from src.models.decoder import Decoder14
        return Decoder14(args.latent_channels, args.decoder_channels, args.channels)
    if args.decoder == '13':
        from src.models.decoder import Decoder13
        return Decoder13(args.r_dim, args.frame_size, args.decoder_channels, args.channels)
    if args.decoder == '12':
        from src.models.decoder import Decoder12
        return Decoder12(args.r_dim, args.frame_size, args.decoder_channels, args.channels, args.batch_norm)
        #return Decoder12(args.r_dim + args.r2_dim, args.frame_size, args.decoder_channels, args.channels, args.batch_norm)
    if args.decoder == '11':
        from src.models.decoder import Decoder11
        return Decoder11(args.r_dim, args.frame_size, args.decoder_channels, args.channels, args.latent_channels)
    if args.decoder == '10':
        from src.models.decoder import Decoder10
        return Decoder10(args.r_dim, args.frame_size, args.decoder_channels, args.channels, args.latent_channels)
    if args.decoder == '9':
        from src.models.decoder import Decoder9
        return Decoder9(args.r_dim, args.frame_size, args.decoder_channels, args.channels, args.latent_channels)
    if args.decoder == '9NoSig':
        from src.models.decoder import Decoder9NoSig
        return Decoder9NoSig(args.r_dim, args.frame_size, args.decoder_channels, args.channels, args.latent_channels)
    if args.decoder == '8':
        from src.models.decoder import Decoder8
        return Decoder8(args.r_dim, args.frame_size, args.decoder_channels, args.channels, args.latent_channels)
    if args.decoder == '8NoSig':
        from src.models.decoder import Decoder8NoSig
        return Decoder8NoSig(args.r_dim, args.frame_size, args.decoder_channels, args.channels, args.latent_channels)

def convert_labels_to_str(labels):
    if labels is None:
        return 'all'
    else:
        res = ''
        for l in labels:
            res += str(l)
        return res

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

# mnist loading utils
def get_data_paths(args, ds):
    #data_folder = './data/' + args.data_set + '/'
    data_folder = '../Video-Novelty/data/' + args.data_set + '/'
    ds_chosen = ds[args.data_vers]
    #ds_chosen = ds['set24']
    #ds_chosen = ds['set23']
    train_data = data_folder + 'train_' + ds_chosen[0]
    train_labels = data_folder + 'train_' + ds_chosen[1]
    val_data = data_folder + 'val_' + ds_chosen[0]
    val_labels = data_folder + 'val_' + ds_chosen[1]
    results = {"train_data" : train_data,
                "train_labels" : train_labels,
                "val_data" : val_data,
                "val_labels" : val_labels}
    return results

def read_data_sets(fpath):
    lines = open(fpath).read().splitlines()
    result = dict()
    for i in range(2, len(lines) - 2):
        line = lines[i].strip()
        if "set" in line:
            result[line] = []
            for j in range(1, 3):
                result[line].append(lines[i + j].strip())
            i += 1
    return result