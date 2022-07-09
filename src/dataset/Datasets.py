import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


class LorenzDataset(Dataset):

    def __init__(self, args, data_path):
        data = np.load(data_path, allow_pickle=True).item()
        self.x = torch.tensor(data['x']).view(-1, args.timesteps, args.u_dim)
        self.dx = torch.tensor(data['dx']).view(-1, args.timesteps, args.u_dim)

    def __len__(self):
        return self.x.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx]