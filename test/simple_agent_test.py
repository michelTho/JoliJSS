import sys
sys.path.insert(0, '../')

import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from memory import Transition
from simple_agent import SimpleAgent

def test_action_selection():
    simple_agent = SimpleAgent(2, 2, 12, 2, 4, device)
    simple_agent.eps_start = 0
    state = np.random.uniform(0, 5, (2, 6))
    action_agent = simple_agent.select_action(state)
    print(action_agent)
    net_input = torch.tensor(state).view(-1).float()
    print(net_input)
    q_values = simple_agent.policy_net(net_input)
    print(q_values)
    real_action = q_values.argmax().item()
    print(real_action)

def test_store():
    simple_agent = SimpleAgent(2, 2, 12, 2, 4, device)
    simple_agent.store(np.random.uniform(0, 5, (2, 6)),
                        0, -1,
                        np.random.uniform(0, 5, (2, 6)))
    print(simple_agent.memory.sample(1))

def test_train_one_step():
    simple_agent = SimpleAgent(2, 2, 12, 2, 4, device)
    for i in range(3):
        simple_agent.store(np.random.uniform(0, 5, (2, 6)),
                            0, -1,
                            np.random.uniform(0, 5, (2, 6)))
    simple_agent.store(np.random.uniform(0, 5, (2, 6)),
                            0, -1, None)
    transitions = simple_agent.memory.sample(4)
    batch = Transition(*zip(*transitions))
    print(batch)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                        dtype=torch.bool)

    non_final_next_states = torch.cat([
        s for s in batch.next_state if s is not None
    ], dim=1).transpose(0, 1)
    state_batch = torch.cat(batch.state, dim=1).transpose(0, 1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    print(non_final_mask)
    print(non_final_next_states)
    print(state_batch)
    print(action_batch)
    print(reward_batch)
    
    state_action_values = simple_agent.policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(4)
    next_state_values[non_final_mask] = simple_agent.target_net(non_final_next_states).max(1)[0].detach()
    print(state_action_values)
    print(next_state_values)

    expected_state_action_values = (next_state_values * 0.999) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)
    print(expected_state_action_values)


test_action_selection()
print('===============================')
test_store()
print('===============================')
test_train_one_step()
