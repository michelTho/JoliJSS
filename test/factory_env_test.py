import sys
sys.path.insert(0, '../')

import numpy as np

from factory_env import FactoryEnv

def test_check_machine_occupation():
    env = FactoryEnv(
        3, 3, 
        np.array([[0, 1, 2], [1, 1, 1], [2, 1, 0]]), 
        np.array([[10,10,10],[10,10,10],[10,10,10]]),
        encoding='classic',
        time_handling='steps'
    )
    
    assert env.check_machine_occupation() == False

    env.step(0)
    env.step(1)
    env.step(2)
    
    assert env.check_machine_occupation() == True

    for i in range(10):
        env.step(3) 
    
    assert env.check_machine_occupation() == False

    env.step(1)

    assert env.check_machine_occupation() == True

test_check_machine_occupation()
