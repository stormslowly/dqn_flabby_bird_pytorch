import random
from torch.autograd import Variable


class Agent(object):
    def __init__(self, model, action_size):
        self.model = model
        self.action_size = action_size

    def select_action(self, state, eps_threshold=0.05):
        sample = random.random()
        if sample > eps_threshold:
            return self.model.forward(Variable(state, volatile=True)).data.max(1)[1][0]
        else:
            if random.random() > 0.8:
                return 0
            return 1
