from os import path
from env import FlappyEnvironment
import torch
import DQN

import torch.autograd

Variable = torch.autograd.Variable

env = FlappyEnvironment(mem_size=1)

model = DQN.DQN()

if path.exists('./dqn.net'):
    model.load_state_dict(torch.load('./dqn.net'))

env.reset()

state1 = env.get_screen()

action = 1
env.step(action)

state2 = env.get_screen()

print (state1.size())

print(
    model.forward(torch.autograd.Variable(state1)).data
)

Q_s_a = model.forward(Variable(state1)).data[0, action]

r = 0

Q_s_max = model.forward(Variable(state2)).data.max(1)[0][0]

print('diff', Q_s_max - Q_s_a)
