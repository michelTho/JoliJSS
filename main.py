import numpy as np
import pickle
import time
import torch

from factory_env import FactoryEnv
from simple_agent import SimpleAgent

from benchmark import AFFECTATIONS, TIMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_jobs = 2 
n_machines = 2

affectations = np.floor(np.random.uniform(0, n_machines, (n_jobs, n_machines))).astype(np.int32) 
times = np.floor(np.random.uniform(1, 10, (n_jobs, n_machines))).astype(np.int32) 

print(affectations)
print(times)
time.sleep(2)

# affectations = np.array(AFFECTATIONS) - 1
# times = np.array(TIMES)

hidden_size = 8

env = FactoryEnv(n_jobs, n_machines, affectations, times) 
agent = SimpleAgent(n_jobs, 
                    n_machines, 
                    env.get_state_space_dimension(), 
                    env.get_action_space_dimension(),
                    hidden_size,
                    device)

n_episodes = 5000 
sum_steps = 0

min_achieved_boundary = 2 * np.sum(times)  # Take a high value, just in case

for i in range(n_episodes):
    env.reset()
    state = env.get_state()
    done = False
    n_steps = 0
    actions_taken = []
    values = []

    while not done:
        # action = int(np.floor(np.random.uniform(-1, n_jobs)))  # Random agent
    
        # Take an action w.r.t the agent policy
        action = agent.select_action(state)

        # Compute the value of the taken action
        value = agent.policy_net(agent.convert_state_to_net_input(state))[action].item()

        # Get the resulting reward and next state from environment
        next_state, reward, done, _ = env.step(action)
       
        if done:
            next_state = None

        # Store the (s, a, r, sp) quadruplet for training
        agent.store(state, action, reward, next_state)

        # Set next_state as the new state
        state = next_state

        # Make one agent training step
        agent.train_one_step()

        #if n_steps % 200 == 0:
        #    env.render(verbosity=0)
        
        n_steps += 1
        actions_taken.append(action)
        values.append(value)

    sum_steps += n_steps
    min_achieved_boundary = min(min_achieved_boundary, n_steps)
    print(f"Job done in {n_steps * env.time_step} units of time")
    print(f"Average time : {sum_steps / (i + 1)} steps")
    print(actions_taken)
    print(values)

#env.render()
print(f"Job done in {n_steps * env.time_step} units of time")
print(f"Minimum boudary for time : {np.max(np.sum(times, axis=1))}")
print(f"Minimum achieved boudary for time : {min_achieved_boundary}")
print(f"Maximum boudary for time : {np.sum(times)}")

pickle.dump(agent, open("./simple_agent_save.pickle", "wb"))
