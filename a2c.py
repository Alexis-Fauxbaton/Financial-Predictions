from data_processing import *
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class PPO(nn.Module):
    def __init__(self, input_size, output_size):
        super(PPO, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, 8*input_size),
            nn.ReLU(),
            nn.Linear(8*input_size, 4*input_size),
            nn.ReLU(),
            nn.Linear(4*input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, output_size),
            nn.Softmax(),
        )
        
        self.critic = None







device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")