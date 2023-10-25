import torch
from functools import partial

def xform_vis(x, vmax=1, eps=1e-7, sin_out=False):
    eps=torch.as_tensor(eps,device=x.device)
    vmax=torch.as_tensor(vmax,device=x.device)
    y0 = torch.log10(eps)
    y1 = torch.log10(vmax+ eps)

    y = torch.log10(x + eps)
    y -= y0
    y /= (y1 - y0)

    if sin_out:
        return 2*y - 1

    return y


def inv_xform_vis(y, vmax=1, eps=1e-7, sin_out=False):
    eps=torch.as_tensor(eps,device=y.device)
    vmax=torch.as_tensor(vmax,device=y.device)
    y0 = torch.log10(eps)
    y1 = torch.log10(vmax + eps)

    if sin_out:
        y = (y+1)/2

    x = torch.pow(10., (y * (y1-y0) + y0)) - eps
    return x


def partial_xform_vis(kwargs):

    if isinstance(kwargs,dict):
        xform = partial(xform_vis, **kwargs)
        inv_xform = partial(inv_xform_vis, **kwargs)
        return xform, inv_xform
    else:
        identity = lambda x, lib=None: x
        return identity, identity
