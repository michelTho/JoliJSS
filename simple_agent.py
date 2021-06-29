import math
import random
import time
import torch
import torch.optim as optim
import torch.nn as nn

from simple_dqn_model import DQN
from memory import ReplayMemory, Transition

class SimpleAgent:

    def __init__(self, n_jobs, n_machines, input_size, n_actions, hidden_size, device):
        self.batch_size = 512
        self.gamma = 1 
        self.eps_start = 0.9
        self.eps_end = 0.
        self.eps_decay = 50000
        self.target_update = 1000

        self.n_actions = n_actions
        self.device = device

        self.policy_net = DQN(input_size, n_actions, hidden_size, device)
        self.target_net = DQN(input_size, n_actions, hidden_size, device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.memory = ReplayMemory(100000)

        self.steps_done = 0

        self.is_training = True
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
   
    @classmethod
    def convert_state_to_net_input(cls, state):
        return torch.tensor(state).view(-1).float() # Reshape tensor to make a column

    def get_epsilon(self):
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        return eps_threshold

    def select_action(self, state):
        return self.select_action_id(state)

    def select_action_id(self, state):
        sample = random.random()
        if sample > self.get_epsilon() or not self.is_training:
            with torch.no_grad():
                net_input = self.convert_state_to_net_input(state)
                return self.policy_net(net_input).argmax().item()  # item is here to convert tensor to int
        else:
            return random.randrange(self.n_actions)

    def store(self, state, action, reward, next_state):
        self.memory.push(torch.tensor(state, device=self.device).view(-1, 1).float(), 
                         torch.tensor(action, device=self.device).view(-1, 1), 
                         torch.tensor([reward], device=self.device), 
                         torch.tensor(next_state, device=self.device).view(-1, 1).float()
                             if next_state is not None else None) 

    def train_one_step(self):
        if len(self.memory) < self.batch_size or not self.is_training:
            return 0
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                        device=self.device,
                                        dtype=torch.bool)
        
        non_final_next_states = torch.cat([
            s for s in batch.next_state if s is not None
        ], dim=1).transpose(0, 1)
        state_batch = torch.cat(batch.state, dim=1).transpose(0, 1)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss_value = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss_value

    def train(self):
        self.is_training = True

    def eval(self):
        self.is_training = False
