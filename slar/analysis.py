import torch

def vis_bias(target : torch.Tensor, pred : torch.Tensor, threshold : float = 0.):
    '''
    Function to compute the visibility bias (the mean of 2 * |target - pred| / (target + pred))

    Parameters
    ----------
    target : torch.Tensor
        The reference visibility based on which the bias is calculated.
    pred : torch.Tensor
        The subject visibility for which the bias is calculated.
    threshold : float
        The visibility lowest threshold. The visibility bias is computed only
        for the instances for which the reference (target) tensor contains the
        visibility value above this threshold.

    Returns
    -------
    torch.Tensor
        The model visibility bias.
    '''
    mask = target > threshold
    a = pred[mask]
    b = target[mask]
    bias = (2 * torch.abs(a-b) / (a+b)).mean()
    return bias