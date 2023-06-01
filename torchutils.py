import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TSDataset(Dataset):
    
    def __init__(self, data) -> None:
        super().__init__()
        
        self.data = data.copy()
        self.unix = data["Unix"]
        self.data.drop("Unix", axis=1, inplace=True)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data.loc[index]