import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


class FeatureFlatten(nn.Module):
    def __init__(self):
        super(FeatureFlatten, self).__init__()

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def __repr__(self):
        return 'FeatureFlatten()'


class VariableSizeInspector(nn.Module):
    def __init__(self):
        super(VariableSizeInspector, self).__init__()

    def forward(self, x):
        print('x size', x.size())
        return x

    def __repr__(self):
        return 'VariableSizeInspector()'


def tensor_image_to_numpy_image(t):
    return t.squeeze(0).permute(1, 2, 0).numpy()


numpy_image_to_tensor_image = T.Compose([
    T.ToPILImage(),
    T.Scale((108, 192), interpolation=Image.CUBIC),
    T.ToTensor(),
])
