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
    if target.shape != pred.shape:
        raise ValueError(f'target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}')
    
    mask = target > threshold
    a = pred[mask]
    b = target[mask]
    bias = (2 * torch.abs(a-b) / (a+b)).mean()
    return bias

def abs_bias(target: torch.Tensor, pred : torch.Tensor, random=0):
    '''
    Function to compute the absolute bias (the mean of |target - pred|)
    
    Parameters
    ----------
    target : torch.Tensor
        Some reference target based on which the bias is calculated.
    pred : torch.Tensor
        Prediction for target on which which the bias is calculated.
        
    Returns
    -------
    torch.Tensor
        The model absolute bias.

    '''
    if target.shape != pred.shape:
        raise ValueError(f'target and pred must have the same shape {(*target.shape,)} != {(*pred.shape,)}')
    
    return torch.abs(target - pred).mean()