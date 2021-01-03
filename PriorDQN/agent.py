import random
import math
import numpy as np

from memory import *
from net import DQN

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as T


class Agent:
    def __init__(self, gamma, lr, input_dims, n_actions, mem_size, batch_size,
                  replace, device):

        self.gamma = gamma
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        
        self.memory = ReplayMemory(mem_size)
        self.td_error_memory = TDErrorMemory(mem_size)
        
        self.batch_size = batch_size
        self.replace = replace
        self.device = device

        self.main_net = DQN(n_actions=self.n_actions).to(self.device)
        self.target_net = DQN(n_actions=self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.main_net.state_dict())

        print(self.main_net)

        self.optimizer = optim.Adam(self.main_net.parameters(), lr=lr)

        self.n_steps = 0

    def decide_action(self, state, mode='train'):
        """
        :param steps: 用来更新eposilon的步数, 可以是episode
        :type steps: int
        """
        if mode == 'train':
            self.n_steps += 1
            if self.n_steps % self.replace == 0:
                self.update_target()

        epsilon = 0.5 * (1 / (self.n_steps * 0.1 + 1))

        if epsilon <= np.random.uniform(0, 1) or mode == 'test':
            self.main_net.eval()
            with torch.no_grad():
                action = self.main_net(state.to('cuda')).max(1)[1].view(1, 1)

        else:
            action = torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)

        return action

    def update(self):
        """
        经验回放更新网络参数
        """
        # 检查经验池数据量是否够一个批次
        if len(self.memory) < self.batch_size:
            return

        # 创建小批量数据
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 找到
        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main()

    def make_minibatch(self):
        """创建小批量数据"""

        # transitions = self.memory.sample(self.batch_size)
        indexes = self.td_error_memory.get_prioritized_indexes(self.batch_size)
        transitions = [self.memory.memory[x] for x in indexes]

        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward)))

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        return batch, state_batch, action_batch, reward_batch, non_final_next_states
    
    def make_batch_of(self, indexes):
        transitions = [self.memory.memory[x] for x in indexes]

        batch = Transition(*zip(*transitions))

        actions = tuple((map(lambda a: torch.tensor([[a]], device=self.device), batch.action)))
        rewards = tuple((map(lambda r: torch.tensor([r], device=self.device), batch.reward)))

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(self.device)

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(actions)
        reward_batch = torch.cat(rewards)

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        """获取期望的Q值"""

        self.main_net.eval()
        self.target_net.eval()

        self.state_action_values = self.main_net(self.state_batch).gather(1, self.action_batch)

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, self.batch.next_state)),
            device=self.device, dtype=torch.bool)

        next_state_values = torch.zeros(self.batch_size, device=self.device)

        next_state_values[non_final_mask] = self.target_net(self.non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + self.reward_batch
        return expected_state_action_values

    def update_main(self):
        """更新网络参数"""

        # 将网络切换训练模式
        self.main_net.train()

        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()

        loss.backward()

        for param in self.main_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.main_net.state_dict())

    def save_model(self, filename):
        torch.save(self.main_net, filename)

    def load_model(self, filename):
        self.main_net = torch.load(filename)
        self.target_net.load_state_dict(self.main_net.state_dict())
    
    def memorize(self, state, action, state_next, reward):
        self.memory.push(state, action, state_next, reward)
    
    def memorize_td_error(self, td_error):
        self.td_error_memory.push(td_error)

    def update_td_error_memory(self):
        # 检查内存容量是否足够计算
        if len(self.memory) < self.batch_size:
            return

        for i in range(0, len(self.memory), self.batch_size):
            start, tail = i, min(len(self.memory), i + self.batch_size)
            start = tail - self.batch_size

            # 创建批量数据
            self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_batch_of(list(range(start, tail)))

            # 计算TD误差
            self.expected_state_action_values = self.get_expected_state_action_values()
            td_errors = self.expected_state_action_values - self.state_action_values

            # TD 误差更新
            self.td_error_memory.memory[start:tail] = td_errors.detach().to('cpu').numpy().tolist()[0]
            # if (i == 0): 
            #     print(self.td_error_memory.memory[start:tail])