import argparse
from copy import copy
from datetime import datetime
import time                                                                             
            
import gnuplotlib as gp
import matplotlib.pyplot as plt
import numpy as np                                                                      
import torch   
                                                                                        
from factory_env import FactoryEnv                                                      
from simple_agent import SimpleAgent        
from utils import smooth_curve

parser = argparse.ArgumentParser(description="Compute the values with value iteration"
"algorithm, then compare DQN with these values")
parser.add_argument("n_jobs", help="Number of jobs for the scheduling problem",
                    type=int)
parser.add_argument("n_machines", help="Number of machines for the scheduling problem",
                    type=int)
parser.add_argument("--n_episodes", help="Number of episodes for the training of DQN",
                    type=int, default=3000)
args = parser.parse_args()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                   
                                                                                        
    n_jobs = args.n_jobs                                                                              
    n_machines = args.n_machines                                                                    

    affectations = np.floor(np.random.uniform(0, n_machines, 
        (n_jobs, n_machines))).astype(np.int32)
    times = np.floor(np.random.uniform(1, 10, (n_jobs, n_machines))).astype(np.int32)       

    print(affectations)                                                                     
    print(times)                                                                            
    time.sleep(2)                                                                           

    encoding = 'classic'                                                                    
    time_handling = 'steps'                                                                 

    gamma = 0.9                                                   

    #==================================================================================
    #================================== VALUE ITERATION ===============================
    #==================================================================================

    print("Computing value iteration")
    timer = time.time()

    n_iter = 100                                                                             
                                                                                        
    env = FactoryEnv(n_jobs, n_machines, affectations, times, encoding, time_handling)      
                                                                                        
    states = env.get_state_space()
    actions = env.get_action_space()                                                        
    vi_values = {state: 0 for state in states}                                                 
    
    print(f"There are {len(states)} states")
                                                                                        
    for i in range(n_iter):                                                                 
        if i%(n_iter//10) == 0:
            print(f"Iteration {i}/{n_iter}")
        prev_values = copy(vi_values)                                                          
        for state, value in vi_values.items():                                                 
            possible_values = []                                                            
            if state != env.get_hashable_state(env.get_terminal_state()):                   
                for action in actions:                                                      
                    next_state, reward, done, _ = env.try_state_action_pair(                
                        env.get_array_state(state), action)                                 
                    next_state = env.get_hashable_state(next_state)                         
                    possible_values.append(reward + gamma * prev_values[next_state])        
                vi_values[state] = max(possible_values)                                        
                                                                                        
    print(f"Value iteration computed in {time.time() - timer: .1f} sec")

    #==================================================================================
    #======================================= DQN ======================================
    #==================================================================================

    print("Computing DQN")
    timer = time.time()

    hidden_size = 256                                                                       
                                                                                        
    agent = SimpleAgent(n_jobs + 1,                                                         
                        n_machines,                                                         
                        env.get_state_space_dimension(),                                    
                        env.get_action_space_dimension(),                                   
                        hidden_size,                                                        
                        device)                                                             
                                                                                        
    n_episodes = args.n_episodes                                                                
                                                                                        
    rewards = []                                                                            
    n_steps_tab = []
    squared_difs = []
    dqn_values = {state: 0 for state in states}

    for i in range(n_episodes):                                                             
        state = env.reset()                                                                 
        done = False                                                                        
                                                                                        
        while not done:                                                                     
            # Take an action w.r.t the agent policy                                         
            action = agent.select_action(state)                                             
                                                                                        
            # Compute the value of the taken action                                         
            value = agent.policy_net(agent.convert_state_to_net_input(state)
                )[action].item()
                                                                                        
            # Get the resulting reward and next state from environment                      
            next_state, reward, done, info = env.step(action)                               
            rewards.append(reward)                                                          
                                                                                        
            if done:                                                                        
                next_state = None                                                           
                                                                                        
            # Store the (s, a, r, sp) quadruplet for training                               
            agent.store(state, action, reward, next_state)                                  
                                                                                        
            # Set next_state as the new state                                               
            state = next_state                                                              
                                                                                        
            # Make one agent training step                                                  
            agent.train_one_step()                                           
                                                                                        
            n_steps = info["n_steps"]

        n_steps_tab.append(n_steps)
        squared_dif = 0
        for state, value in dqn_values.items():
            action = agent.select_action(env.get_array_state(state))
            value = agent.policy_net(agent.convert_state_to_net_input(
                env.get_array_state(state)))[action].item()
            dqn_values[state] = value
            squared_dif += (value - vi_values[state])**2
        squared_difs.append(squared_dif)
        
        if i % (n_episodes//10) == 0:
            print(f"Episode {i}/{n_episodes}")
            # We print here an histogram to show how similar the distrib of values are
            differences = {i: 0 for i in range(20)}
            for s, v in dqn_values.items():
                differences[min(int(np.abs(v - vi_values[s])), 19)] += 1
            for i, d in differences.items():
                print(i, d)

    # And finally, we write all the resuts of the experiment in a file
    filename = "experiments/" + str(n_jobs) + "jobs" + str(n_machines) + "machines-" +\
                datetime.now().strftime("%d-%m-%y-%H-%M")
    f = open(filename + ".txt", "w+")
    f.write(f"Experience {datetime.now().strftime('%d/%m/%y %H:%M')}\n")
    f.write(f"Number of jobs : {n_jobs}\n")
    f.write(f"Number of machines : {n_machines}\n")
    f.write(f"Number of states : {len(states)}\n")
    differences = {i: 0 for i in range(20)}
    for s, v in dqn_values.items():
        differences[min(int(np.abs(v - vi_values[s])), 19)] += 1
    for i, d in differences.items():
        f.write(str(i) + ", " + str(d) + "\n")
    
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6,9))
    ax[0].plot(squared_difs)
    ax[1].plot(smooth_curve(rewards, 0.99))
    ax[2].plot(n_steps_tab)
    plt.savefig(filename + ".png")


if __name__ == "__main__":
    main(args)
