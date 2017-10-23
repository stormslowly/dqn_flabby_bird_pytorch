from __future__ import print_function

from itertools import count

import gym
from os import path
from torch.autograd import Variable
import gym_ple
import DQN
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import math

from matplotlib import pyplot as plt

from Agent import Agent
from RelayMemory import ReplayMemory, Transition
from env import FlappyEnvironment

from utilenn import tensor_image_to_numpy_image, numpy_image_to_tensor_image

GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

env = FlappyEnvironment()

model = DQN.DQN()

if path.exists('./dqn.net'):
    model.load_state_dict(torch.load('./dqn.net'))

agent = Agent(model, 2)

plt.figure(1)
env.reset()

print('state_size ', env.current_state.size())
img = plt.imshow(tensor_image_to_numpy_image(env.current_state))

optimizer = optim.Adam(model.parameters(), lr=0.05)

total_loss = []

plt.figure(2)
plt.plot(total_loss)


def _optimize_model(memory):
    if len(memory) < BATCH_SIZE:
        return 0

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    next_state = Variable(torch.cat(batch.next_state))

    state_action_values = model(state_batch).gather(1, action_batch)

    # # Compute V(s_{t+1}) for all next states.
    # next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.Tensor))
    # next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    next_state_values = model(next_state).max(1)[0]

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    # Compute the expected Q values
    # reward_batch.data.clamp_(-1, 1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.data[0]


def optimize_model(memory):
    if len(memory) < BATCH_SIZE:
        return 100

    losses = []

    for i in count():
        loss = _optimize_model(memory)
        losses.append(loss)

        mean_loss = np.mean(losses)
        print(i, mean_loss)

        if i >= 100:
            break
        if i >= 50 and mean_loss < 0.2:
            break

    return np.mean(losses)


current_loss = 2

BATCH_SIZE = 200

for _ in range(10000):

    for c in count():

        env.reset()

        while True:
            action = agent.select_action(env.current_state, math.atan(current_loss * 3) / math.pi * 2)
            done = env.step(action)

            if done:
                break

            img.set_data(tensor_image_to_numpy_image(env.current_state))
            plt.figure(1, figsize=(300, 300))
            plt.draw()
            plt.pause(0.001)

        if c >= 20:
            break

    print('epoch ', _)

    current_loss = optimize_model(env.mem)
    total_loss.append(current_loss)

    plt.figure(2, figsize=(500, 500))
    plt.gcf().gca().cla()
    plt.plot(np.log10(total_loss))
    plt.pause(0.01)

    torch.save(model.state_dict(), './dqn.net')
