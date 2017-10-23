from torch import nn
from torch import functional as F
from torch.autograd import Variable
import torch

from utilenn import VariableSizeInspector, FeatureFlatten


class CNNWithBatchNormalReLU(nn.Module):
    def __init__(self, **kwargs):
        super(CNNWithBatchNormalReLU, self).__init__()

        in_channels = kwargs.get('in_channels', 3)
        out_channels = kwargs.get('out_channels', 5)
        kernel_size = kwargs.get('kernel_size', 5)
        stride = kwargs.get('stride', 2)

        bn_num_features = kwargs.get('bn_num_features', 16)

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cnn(x)

    def __repr__(self):
        return 'CNNWithBatchNormalReLU()'


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = nn.Sequential(
            CNNWithBatchNormalReLU(in_channels=3,
                                   out_channels=5,
                                   kernel_size=4,
                                   bn_num_features=16),

            CNNWithBatchNormalReLU(in_channels=5,
                                   out_channels=10,
                                   kernel_size=6,
                                   bn_num_features=5),

            CNNWithBatchNormalReLU(in_channels=10,
                                   out_channels=3,
                                   kernel_size=10,
                                   bn_num_features=5),
            FeatureFlatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(147, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        return self.head(x.view(x.size(0), -1))


if __name__ == '__main__':
    net = DQN()
    # print(net)
    import time

    start = time.time()

    data = net.forward(Variable(torch.rand(2, 3, 100, 100))).data

    print('the max ', data)
    print('index', data.max(1)[0])
    print(time.time() - start)
