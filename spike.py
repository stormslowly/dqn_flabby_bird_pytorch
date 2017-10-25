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
from RelayMemory import Transition
from env import FlappyEnvironment

from utilenn import tensor_image_to_numpy_image, numpy_image_to_tensor_image

env = FlappyEnvironment()

model = DQN.DQN()

if path.exists('./dqn.net'):
    model.load_state_dict(torch.load('./dqn.net'))

agent = Agent(model, 2)

env.reset()

optimizer = optim.Adam(model.parameters(), lr=0.001)

total_loss = []


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

        if i >= 50:
            break
        if i >= 30 and mean_loss < 1:
            break
    torch.save(model.state_dict(), './dqn.net')

    return np.mean(losses)


current_loss = 1

initial_epsilon = 1.
final_epsilon = 0.1
GAMMA = 0.99

exploration = 500
mem_size = 5000
BATCH_SIZE = 128

delta = (initial_epsilon - final_epsilon) / exploration


def test_result(epoch):
    best_step = 0
    for _ in range(3):

        env.reset()
        step = 0
        while True:
            action = agent.select_action(
                env.current_state,
                -1
            )
            done = env.step(action)
            step += 1

            if done:
                break
        best_step = max(best_step, step)

    print(epoch, 'best step ', best_step)


for epoch in range(100):

    epsilon = initial_epsilon

    for c in range(exploration):
        env.reset()

        epsilon -= delta
        while True:
            action = agent.select_action(
                env.current_state,
                epsilon
            )
            done = env.step(action)

            if done:
                break

        print('loss', epsilon, optimize_model(env.mem))

        test_result(epoch)
