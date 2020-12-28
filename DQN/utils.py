import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(reward, filename='reward.png', dirname='assets/', lines=None):
    fig=plt.figure()

    x = [i+1 for i in range(len(reward))]

    plt.plot(x, reward, color="r")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.grid(color="k", linestyle=":")
    plt.savefig(dirname + filename)