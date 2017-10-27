import torch.nn as nn
import torchvision.transforms as T
import torch
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


toPIL = T.ToPILImage()


def tensor_image_to_numpy_image(t):
    return toPIL(t.cpu().squeeze(0))


def to_gray_pil(pil_image):
    return pil_image.convert('L')


def tryCuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


numpy_image_to_tensor_image = T.Compose([
    T.ToPILImage(),
    T.Scale((40, 40), interpolation=Image.HAMMING),
    to_gray_pil,
    T.ToTensor(),
    tryCuda
])
