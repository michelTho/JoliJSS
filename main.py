import numpy as np
import time
import torch

from factory_env import FactoryEnv
from simple_agent import SimpleAgent

from benchmark import AFFECTATIONS, TIMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_jobs = 15 
n_machines = 15

# affectations = np.floor(np.random.uniform(0, n_machines, (n_jobs, n_machines))).astype(np.int32) 
# times = np.floor(np.random.uniform(1, 10, (n_jobs, n_machines))).astype(np.int32) 
affectations = np.array(AFFECTATIONS) - 1
times = np.array(TIMES)

env = FactoryEnv(n_jobs, n_machines, affectations, times) 
agent = SimpleAgent(n_jobs, 
                    n_machines, 
                    env.get_state_space_dimension(), 
                    env.get_action_space_dimension(), 
                    device)
agent.eval()

n_episodes = 2

timer1 = 0
timer2 = 0
timer3 = 0
timer4 = 0
cur_time = time.time()

for i in range(n_episodes):
    env.reset()
    state = env.get_state()
    done = False
    n_steps = 0

    while not done:
        # action = int(np.floor(np.random.uniform(-1, n_jobs)))  # Random agent
    
        # Take an action w.r.t the agent policy
        action = agent.select_action(state)
        timer1 += time.time() - cur_time
        cur_time = time.time()

        # Get the resulting reward and next state from environment
        next_state, reward, done, _ = env.step(action)
        timer2 += time.time() - cur_time
        cur_time = time.time()
        
        if done:
            next_state = None

        # Store the (s, a, r, sp) quadruplet for training
        agent.store(state, action, reward, next_state)
        timer3 += time.time() - cur_time
        cur_time = time.time()
        
        # Set next_state as the new state
        state = next_state
    
        # Make one agent training step
        agent.train_one_step()
        timer4 += time.time() - cur_time
        cur_time = time.time()
        
        if n_steps % 200 == 0:
            env.render(verbosity=0)
        
        n_steps += 1

print(timer1)
print(timer2)
print(timer3)
print(timer4)

env.render()
print(f"Job done in {n_steps * env.time_step} units of time")
print(f"Minimum boudary for time : {np.max(np.sum(times, axis=1))}")
print(f"Maximum boudary for time : {np.sum(times)}")
