import functional as Func
import torch.nn as nn

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return Func.mish(input)