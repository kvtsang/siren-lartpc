import torch
from functools import partial
import numpy as np

def xform_vis(x, vmax=1, eps=1e-7, sin_out=False):
    r'''
    Function to transform a linear visibility value to a log-scale.
    The log-scale is defined in the range 0 to 1.0. These edge values correspond to the
    range limit [eps,vmax] in the linear scale. 

    .. math::
        y = (log10(x+eps) - log10(eps)) / (log10(vmax+eps) - log10(eps))

    If sin_out is True, this is shifted and scaled to fit [-1,+1].

    .. math::
        y = 2*y - 1


    Parameters
    ----------
    x : torch.Tensor
        Input target (a linear-scale visibility value)
    vmax : float
        The maximum value of the target parameter (1.0 for visibility).
    eps : float
        The lowest value to represent in the log scale (0. will become inf) for numerical stability
    sin_out : bool
        If True, the range of a log-scale value would be made [-1,+1]. If False, it is [0,+1].

    Returns
    -------
    torch.Tensor
        The log-scale visibility value.
    '''
    # eps=torch.as_tensor(eps,device=x.device)
    # vmax=torch.as_tensor(vmax,device=x.device)
    y0 = np.log10(eps)
    y1 = np.log10(vmax+ eps)

    y = torch.log10(x + eps)
    y -= y0
    y /= (y1 - y0)

    if sin_out:
        return 2*y - 1

    return y


def inv_xform_vis(y, vmax=1, eps=1e-7, sin_out=False):
    '''
    Function to inverse-transform a log-scale visibility value to a linear scale.

    Parameters
    ----------
    y : torch.Tensor
        The visibility value in log-scale.
    vmax : float
        The vmax (see comments for the xform_vis function) given for the transorm function used to produce y.
    eps : float
        The eps (see comments for the xform_vis function) given for the transform function uased to produce y.
    sin_out : bool
        The sin_out (see commaents for the xform_vis function) given for the transform function used to produce y.

    Returns
    -------
    torch.Tensor
        The linear-scale visibility value.
    '''
    # eps=torch.as_tensor(eps,device=y.device)
    # vmax=torch.as_tensor(vmax,device=y.device)
    y0 = np.log10(eps)
    y1 = np.log10(vmax + eps)

    if sin_out:
        y = (y+1)/2

    x = torch.pow(10., (y * (y1-y0) + y0)) - eps
    return x


def partial_xform_vis(kwargs):
    '''
    Function to create partial function instances.
    '''

    if isinstance(kwargs,dict):
        xform = partial(xform_vis, **kwargs)
        inv_xform = partial(inv_xform_vis, **kwargs)
        return xform, inv_xform
    else:
        identity = lambda x, lib=None: x
        return identity, identity
