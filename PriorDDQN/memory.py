
from collections import namedtuple
import random
import numpy as np

# 存储经验的数据结构
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

class TDErrorMemory(object):
    """
    优先经验回放方法的TD误差存储池
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0
        
    def push(self, td_error):
        '''插入'''
        # 如果空间未满
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = td_error
        self.index = (self.index + 1) % self.capacity
    
    def __len__(self):
        """返回当前内存长度"""
        return len(self.memory)
        
    def get_prioritized_indexes(self, batch_size, epsilon=0.0001):
        """根据TD误差以概率获得index
        :param epsilon: 微小值
        :type float
        """
        
        # 计算TD误差的总和
        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += epsilon * len(self.memory)

        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        # 通过得到的随机数，并按升序排列
        indexes = []
        idx = 0
        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + epsilon)
                idx += 1

            # 由于计算时使用微小值而导致index超过内存大小的修正
            if idx >= len(self.memory):
                idx = len(self.memory) - 1
            indexes.append(idx)

        return indexes

    def update_td_error(self, updated_td_errors):
        '''TD误差的更新'''
        self.memory = updated_td_errors
    

