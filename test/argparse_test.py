"""argparse模块测试
    命令行选项、参数和子命令解析器
    https://docs.python.org/zh-cn/3.6/library/argparse.html
"""

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Reinforcement Learning Implement')

    parser.add_argument('-l', '--lr', type=float, default=0.0001, help='Learning rate for optimizer')

    parser.add_argument('-N', '--episode', type=int, default=1000, help='Number of episodes for training')

    parser.add_argument('-eps_min', type=float, default=0.1, help='Minimum value for epsilon in epsilon-greedy action selection')

    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor for update equation.')

    parser.add_argument('-eps_dec', type=float, default=1e-5, help='Linear factor for decreasing epsilon')

    parser.add_argument('-eps', type=float, default=1.0, help='Starting value for epsilon in epsilon-greedy action selection')

    parser.add_argument('-m', '--memory_capacity', type=int, default=100000, #~13Gb
                                help='Maximum size for memory replay buffer')

    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size for replay memory sampling')

    parser.add_argument('-r', '--replace', type=int, default=1000, help='interval for replacing target network')

    parser.add_argument('-env', '-e', '--environment', type=str, default='PongNoFrameskip-v4',
                            help='Atari environment.\nPongNoFrameskip-v4\n \
                                  BreakoutNoFrameskip-v4\n \
                                  SpaceInvadersNoFrameskip-v4\n \
                                  EnduroNoFrameskip-v4\n \
                                  AtlantisNoFrameskip-v4')

    parser.add_argument('-gpu', type=str, default='0', help='GPU: 0 or 1')

    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='load model checkpoint')

    parser.add_argument('-path', type=str, default='models/',
                        help='path for model saving/loading')

    parser.add_argument('--clip_rewards', type=bool, default=False, help='Clip rewards to range -1 to 1')

    parser.add_argument('-fire_first', type=bool, default=False, help='Set first action of episode to fire')

    args = parser.parse_args()

    fname = 'DDQN' + '_' + args.environment + '_lr' + str(args.lr) + '_ep' + str(args.episode)
    figure_file = 'plots/' + fname + '.png'
    scores_file = fname + '_scores.npy'

    print(fname)
    print(args)

