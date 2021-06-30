import argparse
import random
import time

import gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3 import PPO
import torch

from factory_env import FactoryEnv
from simple_agent import SimpleAgent

from benchmark import AFFECTATIONS, TIMES

parser = argparse.ArgumentParser(description="Compute PPO on FactoryEnv")
parser.add_argument("n_jobs", help="Number of jobs for the scheduling problem",         
                    type=int)                                                           
parser.add_argument("n_machines", help="Number of machines for the scheduling problem", 
                    type=int)                                                           
parser.add_argument("--n_steps", help="Number of episodes for the training of DQN",  
                    type=int, default=int(1e5))
parser.add_argument("--hidden_size", help="size of hidden layers in MlpPolicy",
                    type=int, default=64)
parser.add_argument("--multiprocessing", help="Use this to run on several cores",
                    action='store_true')
args = parser.parse_args()   

def main(args):
    
    seed = 42

    random.seed(seed)
    np.random.seed(seed)

    multiprocessing = args.multiprocessing
    n_envs = 2
    
    n_jobs = args.n_jobs
    n_machines = args.n_machines

    n_steps = args.n_steps

    h_size = args.hidden_size

    policy_kwargs = {
        'activation_fn': torch.nn.ReLU,
        'net_arch': [{'pi':[h_size, h_size], 'vf':[h_size, h_size]}]
    }
    affectations = np.floor(np.random.uniform(0, n_machines, 
        (n_jobs, n_machines))).astype(np.int32) 
    times = np.floor(np.random.uniform(1, 10, (n_jobs, n_machines))).astype(np.int32) 

    print(affectations)
    print(times)
    time.sleep(2)

    # affectations = np.array(AFFECTATIONS) - 1
    # times = np.array(TIMES)
    
    # We need a make_env function to work with SubprocVecEnv
    # This function returns a constructor of the env we want to make
    def make_env(seed):
        def _constructor():
            env = FactoryEnv(n_jobs, n_machines, affectations, times, 
                    encoding='classic', time_handling='steps') 
            env.seed(seed)
            return env
        return _constructor

    if multiprocessing:
        envs = [make_env(seed + i) for i in range(n_envs)]
        env = DummyVecEnv(envs)
    else:
        env = FactoryEnv(n_jobs, n_machines, affectations, times, 
                    encoding='classic', time_handling='steps')     
        check_env(env)  # This doesn't work anymore with VecEnv

    model = PPO(MlpPolicy, env, verbose=1, tensorboard_log="./experiments",
                policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=n_steps, eval_freq=int(1e4), n_eval_episodes=10)
    
    obs = env.reset()
    for i in range(20):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    main(args)
