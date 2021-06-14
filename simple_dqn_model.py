import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, device):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size) 
        self.to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

