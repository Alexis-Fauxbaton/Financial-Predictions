import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)

        self.classifier = nn.Sequential(nn.Linear(hidden_size, int(hidden_size / 2)),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_size / 2),
                                                  int(hidden_size / 4)),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_size / 4), output_size))
        
        self.init_weights()
        
        self.binary = True if output_size == 1 else False

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)

        x = self.classifier(x)

        if not self.binary:
            return F.softmax(x, dim=-1), hidden
            # return x, hidden
        else:
            return F.sigmoid(x), hidden