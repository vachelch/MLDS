import numpy as np
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_loader(train_x, train_y, batch_size):
    batchs = len(train_y)//batch_size
    for i in range(batchs):
        x = train_x[i*batch_size: (i+1)*batch_size]
        y = train_y[i*batch_size: (i+1)*batch_size]
        yield (x, y)
