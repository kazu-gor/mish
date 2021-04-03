import torch
import torch.nn.functional as F

@torch.jit.script
def mish(input: torch.Tensor):
    return input * torch.tanh(F.softplus(input))