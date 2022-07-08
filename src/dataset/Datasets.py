import torch
from torchvision import transforms
import numpy as np
from src.utils.other import get_lorenz_path


class LorenzDataset(Dataset):

    def __init__(self, train=False, val=False, test=False):
        data_paths = get_lorenz_path()
        if train:
            data_path= data_paths[0]
        elif val:
            data_path = data_paths[1]
        else:
            data_path = data_paths[2]
        data = np.load(data_path, allow_pickle=True).item()
        
        self.x = torch.tensor(data['x'])
        self.dx = torch.tensor(data['dx'])

    def __len__(self):
        return self.data.size(0)
    
    def __getitem__(self, idx):
        return self.x[idx], self.dx[idx]