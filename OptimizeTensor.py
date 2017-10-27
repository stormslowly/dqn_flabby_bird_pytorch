import torch

if torch.cuda.is_available():
    OTensor = torch.cuda
else:
    OTensor = torch
