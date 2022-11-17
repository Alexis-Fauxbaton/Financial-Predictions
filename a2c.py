from data_processing import *
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions import Normal
import torch.optim as optim


class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()

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
        )
        
        self.critic = nn.Sequential(
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
            nn.Linear(2*input_size, 1)
        )

    
    def forward(self, x):
        value = self.critic(x)
        #generate logits or probs (if the last actor layer is a softmax) from which we will generate a distribution we will sample from
        logits = self.actor(x)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        return dist, value
    
    def policy_loss(self, old_prob, prob, advantage, eps):
        ratio = prob / old_prob
        
        clip = torch.clamp(ratio, 1-eps, 1+eps)*advantage
        
        loss = torch.min(ratio*advantage, clip)
        
        return -loss

    #TODO
    def critic_loss(self):
        pass


    def train(self, epochs=10, steps_per_epoch = 1000, lr=0.001):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            actions = []
            values = []
            rewards = []
            states = []
            #log probabilities of each action taken on this trajectory
            log_probs = []
            
            
        
        pass



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")