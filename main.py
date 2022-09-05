import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

from src.model import *

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    print("Hi pycharm")

    net = shufflenet_g8_w1()

    X = torch.rand(1, 3, 224, 224)
    X = net(X)
    print(net.__class__.__name__, 'output shape:\t', X.shape)
    assert (tuple(X.size()) == (1, 10))
