import numpy as np

class FactoryEnv:

    def __init__(self, n_jobs, n_machines, affectations, times):
        self.time_step = 1
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.affectations = affectations
        self.times = times
        self.completion = np.zeros((n_jobs, n_machines))
        self.current_jobs = np.zeros((n_jobs, n_machines))
        self.machine_usage = np.zeros(n_machines)

    def step(self, action):
        if action == -1:
            self.take_time_step()
            # We use the same API as gym environments : https://gym.openai.com/docs/
            return self.get_state(), -1, self.check_done(), None
        elif action in range(0, self.n_jobs):
            if np.sum(self.current_jobs[action]) == 0:
                indexes = np.where(self.completion[action] == 0)[0]
                if len(indexes) != 0:
                    index = indexes[0]
                    if self.machine_usage[self.affectations[action][index]] == 0:
                        self.current_jobs[action][index] = 1
            self.take_time_step()
            return self.get_state(), -1, self.check_done(), None
        else: 
            raise Exception(f"The action you provided ({action}) is not valid. "
            f"Please provide an integer between -1 and {self.n_jobs}.")
    
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

    def take_time_step(self):
        # We first complete completion, by adding a time step to running jobs
        # And crop them if they get bigger than completion time
        self.completion = np.minimum(self.completion + self.current_jobs * self.time_step, self.times)
        # Then we check if any job has finished, and update the current_jobs accordingly
        self.current_jobs = np.minimum(self.current_jobs, 1 - 1 * np.equal(self.times, self.completion))
        # And update machine_usage
        self.machine_usage = np.zeros(self.n_machines)
        for i in range(len(self.current_jobs)):
            for j in range(len(self.current_jobs[0])):
                if self.current_jobs[i, j] == 1:
                    self.machine_usage[self.affectations[i, j]] = 1

    def get_state(self):
        # The state of the factory is represented by the concatenation of 3 matrix :
        #  - a first one representing the affectations of the jobs on the machines
        #  - a second one representing the times needed for each job to complete
        #  - and a third one saying how long the job has been running.
        # The third matrix is the only one which is going to change. Once the third matrix
        # Is similar to the second one, all jobs are completed.
        return np.concatenate((self.affectations, self.times, self.completion), axis=0)

    def check_done(self):
        if (self.completion == self.times).all():
            return True
        else: 
            return False

    def reset(self):
        self.completion = np.zeros((self.n_jobs, self.n_machines))
        self.current_jobs = np.zeros((self.n_jobs, self.n_machines))
        self.machine_usage = np.zeros(self.n_machines)

    def get_action_space_dimension(self):
        return self.n_jobs

    def get_state_space_dimension(self):
        return self.n_jobs * self.n_machines * 3
