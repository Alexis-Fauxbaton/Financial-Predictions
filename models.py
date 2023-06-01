import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, output_size) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.classifier = nn.Sequential(nn.Linear(hidden_size, int(hidden_size / 2)),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_size / 2), int(hidden_size / 4)),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_size / 4), output_size))
        
    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        
        x = self.classifier(x)
        
        return torch.softmax(x), hidden