from RelayMemory import ReplayMemory
from utilenn import numpy_image_to_tensor_image, tensor_image_to_numpy_image
import gym
import gym_ple
import torch
import random


class FlappyEnvironment(object):
    def __init__(self):
        self.game = gym.make('FlappyBird-v0')

        self.action_size = 2
        self.current_state = None
        self.game.seed(100)
        self.t = 0

        self.mem = ReplayMemory(10000)

    def reset(self):
        self.t = 0
        self.game.reset()
        self.game.step(1)
        self.current_state = self.get_screen()

    def get_screen(self):
        return numpy_image_to_tensor_image(self.game.render('rgb_array')).unsqueeze(0)

    def step(self, action):
        _state, reward, done, _obs = self.game.step(action)

        next_state = self.get_screen()

        if reward > 0:
            print('nice job', reward)

        self.t += 0.01

        reward += self.t

        if done:
            next_state = self.current_state

        self.current_state = next_state

        if reward <= 0:
            if random.random() > 0.5:
                return done

        self.mem.push(self.current_state, torch.LongTensor([[action]]), next_state, torch.Tensor([reward]), done)

        return done


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    env = FlappyEnvironment()
    env.reset()

    print(env.get_screen().size())
    img = plt.imshow(tensor_image_to_numpy_image(env.get_screen()))

    for i in range(100):
        if env.step(1):
            break

        img.set_data(tensor_image_to_numpy_image(env.get_screen()))
        plt.draw()
        plt.pause(0.1)

    plt.show(block=True)
