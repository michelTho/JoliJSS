import math
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn

from simple_dqn_model import DQN
from memory import ReplayMemory, Transition

class SimpleAgent:

    def __init__(self, n_jobs, n_machines, input_size, n_actions, device):
        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay= 200
        self.target_update = 10

        self.n_actions = n_actions
        self.device = device

        self.policy_net = DQN(input_size, n_actions, device)
        self.target_net = DQN(input_size, n_actions, device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.memory = ReplayMemory(1000000)

        self.steps_done = 0

        self.is_training = True
        self. optimizer = optim.RMSprop(self.policy_net.parameters())
    
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        if sample > eps_threshold or not self.is_training:
            with torch.no_grad():
                state = torch.tensor(state).view(-1).float()  # Reshape tensor to make it a column
                return self.policy_net(state).argmax().item()  # item is here to convert tensor to int
        else:
            return random.randrange(self.n_actions)

    def store(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def train_one_step(self):
        if len(self.memory) < self.batch_size or not self.is_training:
            return
        timer = time.time()
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        print(time.time() - timer)
        timer = time.time()
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                        device=self.device,
                                        dtype=torch.bool)
        
        non_final_next_states = torch.cat([
            torch.tensor(s, device=self.device).view(-1, 1) for s in batch.next_state if s is not None
        ], dim=1).transpose(0, 1).float()
        state_batch = torch.tensor(batch.state, device=self.device).view(self.batch_size, -1).float()
        action_batch = torch.tensor(batch.action, device=self.device).view(-1, 1)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        print(time.time() - timer)
        timer = time.time()
 
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        print(time.time() - timer)
        timer = time.time()
 
        self.optimizer.zero_grad()
        loss.backward()
        print(time.time() - timer)
        timer = time.time()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        print(time.time() - timer)
        timer = time.time()
 
    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False
