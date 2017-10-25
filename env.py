from RelayMemory import ReplayMemory
from utilenn import numpy_image_to_tensor_image, tensor_image_to_numpy_image
import gym
import gym_ple
import torch


class FlappyEnvironment(object):
    def __init__(self, mem_size=5000):
        self.game = gym.make('FlappyBird-v0')

        self.action_size = 2
        self.current_state = None
        self.game.seed(100)
        self.t = 0

        self.mem = ReplayMemory(mem_size)

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

        for _ in range(4):
            if done:
                break
            _state, next_reward, done, _obs = self.game.step(1)  # no tap

            if next_reward != 0:
                reward = next_reward

        print(action, ' =>reward', reward)
        self.current_state = self.get_screen()

        self.mem.push(self.current_state, torch.LongTensor([[action]]), next_state, torch.Tensor([reward]), done)

        return done


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    env = FlappyEnvironment()
    env.reset()
    img = plt.imshow(tensor_image_to_numpy_image(env.get_screen()))

    for _ in range(4):
        print(_)
        img.set_data(tensor_image_to_numpy_image(env.get_screen()))
        env.step(1)
        plt.draw()
        plt.pause(0.1)

    plt.show(block=True)
#
# print(env.get_screen().size())
#
# for i in range(100):
#     if env.step(1):
#         break
#

#
