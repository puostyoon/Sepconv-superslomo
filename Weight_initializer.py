import torch
"""
According to the paper, we should use "convolution aware initialization method" but 
in this code, we use xavier initialization for simplicity."""
def init_weight(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight.data)