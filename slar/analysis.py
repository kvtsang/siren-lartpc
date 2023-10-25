import torch

def vis_bias(target, pred, threshold=0.):
    mask = target > threshold
    a = pred[mask]
    b = target[mask]
    bias = (2 * torch.abs(a-b) / (a+b)).mean()
    return bias