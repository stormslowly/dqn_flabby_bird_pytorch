from __future__ import print_function
import gym
from torch.autograd import Variable
import gym_ple
import DQN
import torch
import numpy as np
import torch.optim as optim

import random

from matplotlib import pyplot as plt
from RelayMemory import ReplayMemory, Transition

from utilenn import tensor_image_to_numpy_image, numpy_image_to_tensor_image

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


class Agent(object):
    def __init__(self, model, action_size):
        self.model = model
        self.action_size = action_size

    def select_action(self, state, eps_threshold=EPS_END):
        sample = random.random()
        if sample > -1:
            return self.model.forward(Variable(state.unsqueeze(0), volatile=True)).data.max(1)[1][0]
        else:
            return random.randrange(self.action_size)


class Environment(object):
    def __init__(self, game, action_size):
        self.game = game
        self.action_size = action_size
        self.current_state = None

        self.mem = ReplayMemory(100000)

    def reset(self):
        self.game.reset()
        self.current_state = self.get_screen()

    def get_screen(self):
        return numpy_image_to_tensor_image(self.game.render('rgb_array'))

    def step(self, action):
        _state, reward, done, _obs = self.game.step(action)

        next_state = self.get_screen()

        self.mem.push(self.current_state, action, next_state, reward, done)

        self.current_state = next_state

        return done


game = gym.make('FlappyBird-v0')

env = Environment(game, 2)

model = DQN.DQN()

agent = Agent(model, 2)

game.reset()
img = plt.imshow(game.render('rgb_array'))

env.reset()

print(env.current_state.size())

optimizer = optim.RMSprop(model.parameters())


def optimize_model(memory):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    state_batch = Variable(torch.cat(batch.state))

    state_batch = torch.stack(map(lambda t: t.state, transitions))
    action_batch = map(lambda t: t.action, transitions)

    values = model.forward(Variable(state_batch))

    print(action_batch)
    print('value', values.size())
    print(values)
    print(torch.LongTensor([action_batch]).transpose(0, 1))
    print('===============')
    print(values.select(1, torch.LongTensor([action_batch]).transpose(0, 1)))

    return

    action_batch = Variable(torch.cat(batch.action))

    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # print(state_action_values)

    # Compute V(s_{t+1}) for all next states.
    # next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    # next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # # Now, we don't want to mess up the loss with a volatile flag, so let's
    # # clear it. After this, we'll just end up with a Variable that has
    # # requires_grad=False
    # next_state_values.volatile = False
    # # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    #
    # # Compute Huber loss
    # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    #
    # # Optimize the model
    # optimizer.zero_grad()
    # loss.backward()
    # for param in model.parameters():
    #     param.grad.data.clamp_(-1, 1)
    # optimizer.step()


for _ in range(1):

    env.reset()

    while True:
        action = agent.select_action(env.current_state)

        done = env.step(action)
        img.set_data(tensor_image_to_numpy_image(env.current_state))
        plt.draw()
        if done:
            break

    BATCH_SIZE = 3
    optimize_model(env.mem)
