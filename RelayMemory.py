import random
from collections import namedtuple, deque
import math

import itertools

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], capacity)
        self.good_memory = deque([], capacity)

    def push(self, *args):
        transition = Transition(*args)

        self.memory.append(transition)

        # if transition.reward[0] > -2:
        #     self.good_memory.append(transition)
        # else:
        #     self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(
            self.memory,
            min(batch_size, len(self.memory))
        )

        # to_get = min(len(self.good_memory), batch_size)
        #
        # from_good = random.sample(self.good_memory, to_get)
        #
        # if batch_size - to_get == 0:
        #     from_normal = random.sample(self.memory, min(len(self.memory), batch_size))
        # else:
        #     from_normal = random.sample(self.memory, min(len(self.memory), batch_size - to_get))
        #
        # return random.sample(
        #     from_good + from_normal,
        #     batch_size)

    def __len__(self):
        return len(self.memory)
