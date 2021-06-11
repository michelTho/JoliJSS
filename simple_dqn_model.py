import torch
import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, input_size, output_size, device):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_size) 
        self.softmax = nn.Softmax(dim=0)
        self.to(device)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

