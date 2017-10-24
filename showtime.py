from itertools import count
from os import path

import torch

import DQN
from Agent import Agent
from env import FlappyEnvironment
from  matplotlib import pyplot as plt

from utilenn import tensor_image_to_numpy_image

env = FlappyEnvironment()

model = DQN.DQN()

if path.exists('./dqn.net'):
    model.load_state_dict(torch.load('./dqn.net'))

agent = Agent(model, 2)

env.reset()

plt.figure(1, figsize=(3, 3))
img = plt.imshow(tensor_image_to_numpy_image(env.current_state))

for c in count():

    env.reset()
    while True:
        action = agent.select_action(
            env.current_state,
            -1
        )
        done = env.step(action)
        img.set_data(tensor_image_to_numpy_image(env.current_state))
        plt.draw()
        plt.pause(0.001)

        if done:
            break
