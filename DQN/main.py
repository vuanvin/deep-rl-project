import argparse
from itertools import count
import math
import random
import numpy as np
import time
import torch
import gym

import sys

from wrappers import *
from utils import *
from agent import Agent

class Environment:
    """Pong 游戏执行环境"""

    def __init__(self, args):

        fname = 'DDQN' + '_' + args.environment + '_lr' + str(args.lr) + '_ep' + str(args.episode)
        self.figure_file = fname + '.png'
        self.model_file = args.path + fname

        self.env = gym.make(args.environment)
        self.env = make_env(self.env)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        

        self.agent = Agent(gamma=args.gamma,
                  lr=args.lr,
                  input_dims=self.env.observation_space.shape,
                  n_actions=self.env.action_space.n,
                  mem_size=args.memory_capacity,
                  batch_size=args.batch_size,
                  replace=args.replace,
                  device=self.device
                  )

        self.init_steps = 10000
        self.n_steps = 0

    def obs2state(self, obs):
        """ 
        观察值转换成状态
        """
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self, n_episodes, render=False):
        """
        训练Agent
        """
        mean_rewards = []
        episode_reward = []

        for episode in range(n_episodes):
            obs = self.env.reset()
            state = self.obs2state(obs)

            # 记录 reward
            total_reward = 0.0
            for t in count():
                self.n_steps += 1
                action = self.agent.decide_action(state, mode='train')

                if render:
                    self.env.render()

                obs, reward, done, info = self.env.step(action)

                total_reward += reward
                
                if not done:
                    next_state = self.obs2state(obs)
                else:
                    next_state = None

                reward = torch.tensor([reward], device=self.device)

                self.agent.memorize(state, action.to('cpu'), next_state, reward.to('cpu'))
                state = next_state

                
                if self.n_steps > self.init_steps:
                    self.agent.update()

                if done:
                    break

            if (episode + 1 ) % 20 == 0:
                    print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(self.n_steps, episode+1, n_episodes, total_reward))
            

            # 计算平均 reward
            episode_reward.append(total_reward)
            mean_100ep_reward = round(np.mean(episode_reward[-100:]), 1)
            mean_rewards.append(mean_100ep_reward)
            
        self.env.close()
        plot_learning_curve(mean_rewards, filename=self.figure_file + '.png' )
        
        # 保存模型
        self.agent.save_model(self.model_file)
        return


    def test(self, n_episodes=1, render=True):


        # 加载模型
        self.agent.load_model(self.model_file)

        dir_suffix = str(time.monotonic())
        env = gym.wrappers.Monitor(self.env, './videos/' + dir_suffix)

        for episode in range(n_episodes):
            obs = env.reset()
            state = self.obs2state(obs)
            total_reward = 0.0

            for t in count():
                action = self.agent.decide_action(state, mode='test')
                if render:
                    env.render()
                    time.sleep(0.02)

                obs, reward, done, info = env.step(action)

                total_reward += reward

                if not done:
                    next_state = self.obs2state(obs)
                else:
                    next_state = None

                state = next_state

                if done:
                    print("Finished Episode {} with reward {}".format(episode, total_reward))
                    break

        env.close()
        return

    def run(self, train_episodes, test_episodes, render=True):
        self.train(train_episodes)
        self.test(test_episodes, render)


############################################
################## Main ####################
############################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning Implement')

    parser.add_argument('-m', '--mode', type=str, default='train', help='Mode: train, test or all')

    parser.add_argument('-l', '--lr', type=float, default=0.0001, help='Learning rate for optimizer')

    parser.add_argument('-N', '--episode', type=int, default=1000, help='Number of episodes for training')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for update equation.')

    parser.add_argument('-c', '--memory_capacity', type=int, default=100000, help='Maximum size for memory replay buffer')

    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for replay memory sampling')

    parser.add_argument('-r', '--replace', type=int, default=1000, help='Interval for replacing target network')

    parser.add_argument('-env', '-e', '--environment', type=str, default='PongNoFrameskip-v4',
                            help='Atari environment.\nPongNoFrameskip-v4\n \
                                  BreakoutNoFrameskip-v4\n \
                                  SpaceInvadersNoFrameskip-v4\n \
                                  EnduroNoFrameskip-v4\n \
                                  AtlantisNoFrameskip-v4')

    parser.add_argument('--gpu', type=str, default='0', help='GPU: 0 or 1')

    parser.add_argument('--load_checkpoint', type=bool, default=False, help='load model checkpoint')

    parser.add_argument('-p' ,'--path', type=str, default='models/', help='path for model saving/loading')

    parser.add_argument('--clip_rewards', type=bool, default=False, help='Clip rewards to range -1 to 1')


    args = parser.parse_args()

    pong_env = Environment(args)
    

    if args.mode == 'run':
        pong_env.run(train_episodes=args.episode, test_episodes=1, render=True)

    elif args.mode == 'train':
        pong_env.train(n_episodes=args.episode, render=False)

    elif args.mode == 'test':
        pong_env.test(n_episodes=1, render=True)

    else:
        print("Enter correct mode name please.")