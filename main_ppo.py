import pickle
import random
import time

import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import torch

from factory_env import FactoryEnv
from simple_agent import SimpleAgent

from benchmark import AFFECTATIONS, TIMES

seed = 42

random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_jobs = 3
n_machines = 3

affectations = np.floor(np.random.uniform(0, n_machines, 
    (n_jobs, n_machines))).astype(np.int32) 
times = np.floor(np.random.uniform(1, 10, (n_jobs, n_machines))).astype(np.int32) 

print(affectations)
print(times)
time.sleep(2)

# affectations = np.array(AFFECTATIONS) - 1
# times = np.array(TIMES)

env = FactoryEnv(n_jobs, n_machines, affectations, times, 
                    encoding='classic', time_handling='steps')
check_env(env)

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=int(3e5))

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=100)
print(mean_reward, std_reward)

obs = env.reset()
for i in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
