
from collections import namedtuple
import random

# 
Transition = namedtuple(
    'Transion', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    经验回放的存储池
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        '''插入'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        '''抽样'''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        '''返回当前内存容量'''
        return len(self.memory)


class PrioritizedReplay(object):
    """
    优先经验回放
    """
    def __init__(self, capacity):
        pass