from src.dataset.Datasets import *

def load_data(args):
    # train and val data (using val as "test" data)
    if args.data_set == "lorenz":
        folder, data_paths = get_lorenz_path()
        train_set = LorenzDataset(args, data_paths[0])
        val_set = LorenzDataset(args, data_paths[1])
        test_set = LorenzDataset(args, data_paths[2])
    return train_set, val_set, test_set

def load_model(net, cp_path, device, optim=None, scheduler=None):
    checkpoint = torch.load(cp_path, map_location="cuda:" + str(device))
    net.load_state_dict(checkpoint['model'])
    net.to(device)
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
    initial_e = checkpoint['epoch']
    return net, optim, scheduler, initial_e

def make_model(args):
    if args.model == 'SINDyAE':
        from src.models.SINDyAE import Net
    if args.model == 'SINDyVAE':
        from src.models.SINDyVAE import Net
    return Net(args)

def get_lorenz_path():
    folder = "data/lorenz/"
    return folder, (folder + "train.npy", folder + "val.npy", folder + "test.npy")

def get_general_path(args):
    return args.data_set + "/" + args.model + "/" + args.session_name + "/"

def get_checkpoint_path(args):
    cp_folder = args.model_folder + get_general_path(args)
    return cp_folder + 'checkpoint.pt', cp_folder

def get_args_path(args):
    args_folder = args.model_folder +get_general_path(args)
    return args_folder + "args.txt", args_folder

def get_tb_path(args):
    train_name = args.tensorboard_folder + get_general_path(args) + "train"
    test_name = args.tensorboard_folder + get_general_path(args) + "val"
    return train_name, test_name

def get_experiments_path(args):
    return args.experiments + get_general_path(args)