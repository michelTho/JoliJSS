import time

import gym
import numpy as np

class FactoryEnv(gym.Env):

    def __init__(self, n_jobs, n_machines, affectations, times, encoding, time_handling):
        self.encoding = encoding
        # There are several possible encodings for the state space. 
        # The classical encoding represent the state as the concatenation
        # of 3 matrixes (one for affectations, another for times, and a third one
        # for current completion times
        # The one-hot encoding represent each job with a one hot vector 
        # of the following form : 
        # one hot for i, one hot for j, one hot for affectation, time, and completion
        assert self.encoding in {'classic', 'one-hot'}

        self.time_handling = time_handling
        # There are several ways to handle time : either you make time steps,
        # or you act event based (you only let the agent act when there is 
        # something to do
        assert self.time_handling in {'steps', 'event'}

        self.n_steps = 0

        self.time_step = 1
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        
        self.affectations = affectations
        self.times = times
        
        self.completion = np.zeros((n_jobs, n_machines))
        
        # The current_jobs variable is a (n_jobs, n_machines) matrix filled with
        # O and 1. M(i,j) is 1 if job i is assigned to machine j and 0 otherwise
        self.current_jobs = np.zeros((n_jobs, n_machines))
        
        # The machine_usage variable is a list of 0 and 1.
        # There is a 0 for free machines, and 1 for used machines
        self.machine_usage = np.zeros(n_machines)

        # Gym compatibility
        self.action_space = gym.spaces.Discrete(
            self.get_action_space_dimension()
        )
        self.observation_space = gym.spaces.Box(
             low=0, high=10, shape=self.get_state_space_shape(), dtype=np.int32
        )

    def step(self, action):
        
        unadapted_action_taken = True  # A flag to tell if the action has an effect

        if action == self.n_jobs:
            action = -1

        if action == -1: # This action correspond to noop
            if self.check_machine_occupation():
                unadapted_action_taken = False
        elif action in range(0, self.n_jobs):
            if np.sum(self.current_jobs[action]) == 0:
                indexes = np.where(self.completion[action] == 0)[0]
                if len(indexes) != 0:
                    index = indexes[0]
                    if self.machine_usage[self.affectations[action][index]] == 0:
                        self.current_jobs[action][index] = 1
                        self.machine_usage[self.affectations[action][index]] = 1
                        unadapted_action_taken = False
        else: 
            raise Exception(f"The action you provided ({action}) is not valid. "
            f"Please provide an integer between -1 and {self.n_jobs - 1}.")

        # We only move forward if there is no uselessly free machine
        if self.check_machine_occupation():  
            self.move_forward()
        
        # We use the same API as gym environments : https://gym.openai.com/docs/
        reward = 1000 if self.check_done() else -1
        if unadapted_action_taken:
            reward -= -10
        return self.get_state(), reward, self.check_done(), {"n_steps" : self.n_steps}
         
    def render(self, verbosity=0):
        print("PROBLEM DESCRIPTION")
        print("Affectations of jobs on machines")
        print(self.affectations)
        print(" -" * self.n_machines + " - - - - - -")
        print("Times of completion of jobs")
        print(self.times)
        print(" -" * self.n_machines + " - - - - - -")
        print("Completion percentages : ")
        print(np.round(100 * (self.completion / self.times), 0).astype(np.int32))
        if verbosity == 1:
            print("Current jobs")
            print(self.current_jobs)
            print("Machine usage")
            print(self.machine_usage)
    
    def move_forward(self):
        if self.time_handling == 'steps':
            self.take_time_step()
        elif self.time_handling == 'event':
            while self.check_machine_occupation(): 
                if self.check_done():
                    break
                self.take_time_step()

    def take_time_step(self):
        self.n_steps += 1

        # We first complete completion, by adding a time step to running jobs
        # And crop them if they get bigger than completion time
        self.completion = np.minimum(
            self.completion + self.current_jobs * self.time_step, 
            self.times
        )
        # Then, check if any job has finished, and update the current_jobs accordingly
        self.current_jobs = np.minimum(
            self.current_jobs, 
            1 - 1 * np.equal(self.times, self.completion)
        )
        # And update machine_usage
        self.machine_usage = np.zeros(self.n_machines)
        for i in range(len(self.current_jobs)):
            for j in range(len(self.current_jobs[0])):
                if self.current_jobs[i, j] == 1:
                    self.machine_usage[self.affectations[i, j]] = 1

    def get_state(self):
        if self.encoding == 'classic':
            # The state of the factory is represented by the concatenation of 3 matrix :
            #  - a first one representing the affectations of the jobs on the machines
            #  - a second one representing the times needed for each job to complete
            #  - and a third one saying how long the job has been running.
            # The third matrix is the only one which is going to change. Once the third 
            # matrix is similar to the second one, all jobs are completed.
            return np.concatenate((self.affectations, self.times, self.completion), axis=0)
        elif self.encoding == 'one-hot':
            state = []
            for i in range(self.n_jobs):
                for j in range(self.n_machines):
                    cur_job = [0 for k in range(self.n_jobs)]
                    cur_machine = [0 for k in range(self.n_machines)]
                    cur_affectation = [0 for k in range(self.n_machines)] 
                    cur_job[i] = 1
                    cur_machine[j] = 1
                    cur_affectation[self.affectations[i, j]] = 1
                    cur_time = self.times[i, j]
                    cur_completion = self.completion[i, j]
                    state.append(cur_job + cur_machine + cur_affectation + [cur_time] + [cur_completion])
            return np.array(state)

    def check_done(self):
        if (self.completion == self.times).all():
            return True
        else: 
            return False

    def reset(self):
        self.n_steps = 0
        self.completion = np.zeros((self.n_jobs, self.n_machines))
        self.current_jobs = np.zeros((self.n_jobs, self.n_machines))
        self.machine_usage = np.zeros(self.n_machines)
        return self.get_state()

    def get_action_space_dimension(self):
        return self.n_jobs + 1

    def get_state_space_shape(self):
        if self.encoding == 'classic':
            return (self.n_machines * 3, self.n_jobs)
        elif self.encoding == 'one-hot':
            return ((self.n_jobs + self.n_machines * 2 + 2), self.n_machines * self.n_jobs)
    
    def get_state_space_dimension(self):
        shape = self.get_state_space_shape()
        return shape[0] * shape[1]

    def check_machine_occupation(self):
        # The point of this function is to check if there is a machine
        # which could be used, but which isn't because of the ordrers given
        # by the agent. For that, we have to check if there is a job and free
        # machine which matches, but aren't set together.
        if np.sum(self.machine_usage) == self.n_machines:
            return True

        available_jobs = [i for i in range(self.n_jobs) 
                            if np.max(self.current_jobs[i, :]) != 1
                            and not (self.completion[i, :] == self.times[i, :]).all()
        ]
        usable_machines = [self.affectations[
                            job_index, 
                            np.where(self.completion[job_index, :]  == 0)[0][0]
                            ] for job_index in available_jobs]
        if len(set(usable_machines).intersection(
            list(np.where(self.machine_usage == 0)[0]))) != 0:
            return False
        return True
