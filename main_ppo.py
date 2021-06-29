import pickle
import time

import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import torch

from factory_env import FactoryEnv
from simple_agent import SimpleAgent

from benchmark import AFFECTATIONS, TIMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_jobs = 5
n_machines = 5

affectations = np.floor(np.random.uniform(0, n_machines, 
    (n_jobs, n_machines))).astype(np.int32) 
times = np.floor(np.random.uniform(1, 10, (n_jobs, n_machines))).astype(np.int32) 

print(affectations)
print(times)
time.sleep(2)

# affectations = np.array(AFFECTATIONS) - 1
# times = np.array(TIMES)

hidden_size = 256

env = FactoryEnv(n_jobs, n_machines, affectations, times, 
                    encoding='classic', time_handling='steps')
check_env(env)

# agent = SimpleAgent(n_jobs + 1, 
#                     n_machines, 
#                     env.get_state_space_dimension(), 
#                     env.get_action_space_dimension(),
#                     hidden_size,
#                     device)

agent = PPO(MlpPolicy, env, verbose=1)
agent.learn(total_timesteps=10000, eval_freq=1000)

#rewards = []

#while True:
 #   state = env.reset()
  #  actions_taken = []
   # while not done:
    
        # Take an action w.r.t the agent policy
    #    action,  _states = agent.predict(state)

        # Get the resulting reward and next state from environment
     #   next_state, reward, done, info = env.step(action)
      #  rewards.append(reward)

       # actions_taken.append(action)
