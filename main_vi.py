from copy import copy

import numpy as np

from factory_env import FactoryEnv


n_jobs = 5
n_machines = 5
affectations = np.floor(np.random.uniform(0, n_machines, (n_jobs, n_machines))).astype(np.int32)
times = np.floor(np.random.uniform(1, 10, (n_jobs, n_machines))).astype(np.int32) 

encoding = 'classic' 
time_handling = 'steps'

gamma = 1 

n_iter = 10

env = FactoryEnv(n_jobs, n_machines, affectations, times, encoding, time_handling)

states = env.get_state_space()
actions = env.get_action_space()
values = {state: 0 for state in states}

for i in range(n_iter):
    print(i)
    prev_values = copy(values)   
    for state, value in values.items():
        possible_values = []
        if state != env.get_hashable_state(env.get_terminal_state()):
            for action in actions:
                next_state, reward, done, _ = env.try_state_action_pair(
                    env.get_array_state(state), action)
                next_state = env.get_hashable_state(next_state)
                possible_values.append(reward + gamma * prev_values[next_state])
            values[state] = max(possible_values)

for s, v in values.items():
    print(s, v)
