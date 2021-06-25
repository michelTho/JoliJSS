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
agent.learn(total_timesteps=1000000, eval_freq=1000)

n_episodes = 50000 
sum_steps = 0

min_achieved_boundary = 2 * np.sum(times)  # Take a high value, just in case

losses = []
rewards = []

for i in range(n_episodes):
    state = env.reset()
    done = False
    actions_taken = []
    values = []
    n_steps = 0

    while not done:
        # action = int(np.floor(np.random.uniform(-1, n_jobs)))  # Random agent
    
        # Take an action w.r.t the agent policy
        # action = agent.select_action(state)
        action,  _states = agent.predict(state)
        # Compute the value of the taken action
        # value = agent.policy_net(agent.convert_state_to_net_input(state))[action].item()

        # Get the resulting reward and next state from environment
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)

        if done:
            next_state = None

        # Store the (s, a, r, sp) quadruplet for training
        # agent.store(state, action, reward, next_state)

        # Set next_state as the new state
        state = next_state

        # Make one agent training step
        # losses.append(agent.train_one_step())

        #if n_steps % 200 == 0:
        #    env.render(verbosity=0)
        
        # n_steps = info["n_steps"]
        actions_taken.append(action)
        # values.append(value)

    # sum_steps += n_steps
    # min_achieved_boundary = min(min_achieved_boundary, n_steps)
   
    if i % 3 == 0:
        print("================================")
        # print(f"Job done in {n_steps * env.time_step} units of time")
        # print(f"Average time : {sum_steps / (i + 1)} steps")
        print(f"Action taken : \n{actions_taken}")
        # print(f"Values : \n {['%.2f' % v for v in values]}")
        # print(f"Epsilon : {agent.get_epsilon()}")
        print("================================")
        # print(f"Loss : {np.mean(losses[max(len(losses) - 1000, 0):len(losses)])}")
        print(f"Reward : {np.mean(rewards[max(len(rewards) - 1000, 0):len(rewards)])}")

pickle.dump(agent, open("./simple_agent_save.pickle", "wb"))
