from copy import copy                                                                   
import time                                                                             
                                                                                        
import numpy as np                                                                      
import torch                                                                            
                                                                                        
from factory_env import FactoryEnv                                                      
from simple_agent import SimpleAgent        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                   
                                                                                        
n_jobs = 2                                                                              
n_machines = 2                                                                          

affectations = np.floor(np.random.uniform(0, n_machines, (n_jobs, n_machines))).astype(np.int32)
times = np.floor(np.random.uniform(1, 10, (n_jobs, n_machines))).astype(np.int32)       

print(affectations)                                                                     
print(times)                                                                            
time.sleep(2)                                                                           

encoding = 'classic'                                                                    
time_handling = 'steps'                                                                 


#======================================================================================
#================================== VALUE ITERATION ===================================
#======================================================================================


gamma = 1                                                                               
                                                                                        
n_iter = 10                                                                             
                                                                                        
env = FactoryEnv(n_jobs, n_machines, affectations, times, encoding, time_handling)      
                                                                                        
states = env.get_state_space()                                                          
actions = env.get_action_space()                                                        
vi_values = {state: 0 for state in states}                                                 
                                                                                        
for i in range(n_iter):                                                                 
    print(i)                                                                            
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
                                                                                        
for s, v in vi_values.items():                                                             
    print(s, v) 


#======================================================================================
#========================================== DQN =======================================
#======================================================================================


hidden_size = 256                                                                       
                                                                                        
agent = SimpleAgent(n_jobs + 1,                                                         
                    n_machines,                                                         
                    env.get_state_space_dimension(),                                    
                    env.get_action_space_dimension(),                                   
                    hidden_size,                                                        
                    device)                                                             
                                                                                        
n_episodes = 5000                                                                
sum_steps = 0                                                                           
                                                                                        
min_achieved_boundary = 2 * np.sum(times)  # Take a high value, just in case            
                                                                                        
losses = []                                                                             
rewards = []                                                                            

dqn_values = {state: 0 for state in states}

for i in range(n_episodes):                                                             
    state = env.reset()                                                                 
    done = False                                                                        
    actions_taken = []                                                                  
    values = []                                                                         
    n_steps = 0                                                                         
                                                                                        
    while not done:                                                                     
        # Take an action w.r.t the agent policy                                         
        action = agent.select_action(state)                                             
                                                                                        
        # Compute the value of the taken action                                         
        value = agent.policy_net(agent.convert_state_to_net_input(state))[action].item()
                                                                                        
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
        losses.append(agent.train_one_step())                                           
                                                                                        
        n_steps = info["n_steps"]                                                       
        actions_taken.append(action)                                                    
        values.append(value)                                                            
                                                                                        
    sum_steps += n_steps 
    min_achieved_boundary = min(min_achieved_boundary, n_steps)

    if i % 10 == 0:
        print("================================")
        # print(f"Job done in {n_steps * env.time_step} units of time")
        # print(f"Average time : {sum_steps / (i + 1)} steps")
        print(f"Action taken : \n{actions_taken}")
        # print(f"Values : \n {['%.2f' % v for v in values]}")
        print("================================")
        print(f"Reward : {np.mean(rewards[max(len(rewards) - 1000, 0):len(rewards)])}")
        squared_dif = 0
        for state, value in dqn_values.items():
            action = agent.select_action(env.get_array_state(state))
            value = agent.policy_net(agent.convert_state_to_net_input(env.get_array_state(state)))[action].item()
            dqn_values[state] = value
            squared_dif += (value - vi_values[state])**2
        print(f"Root mean squared dif between value iteration values and dqn values : {squared_dif ** 0.5}")
        for s, v in dqn_values.items():
            print(s, v - vi_values[s])
    


