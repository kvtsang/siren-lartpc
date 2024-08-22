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

class RescaleTransform(torch.nn.Module):
    ''' 
    Rescale the range of tensor from `[vmin,vmax]` to `[0,1]` or `[-1,1]`.

    Arguments
    ---------
    vmin: float | torch.Tensor
        Lower bound of the input. It maps to 0 or -1 (when `sin_out = True`).
    vmax : float | torch.Tensor
        Upper bound of the input. It maps to 1.
    sin_out: bool (optional)
        Rescale to `[-1,1]` instead of `[0,1]` (default: False).

    
    Example
    -------
        >>> rescale = RescaleTransform(vmin=-5, vmax=5)
        >>> x = torch.linspace(-5, 5, 21)
        tensor([-5.0, -4.5, -4.0, -3.5, ..., 5.0])
        >>> rescale(x)
        tensor([0.00, 0.05, 0.10, 0.15, ..., 1.00])
    '''
    def __init__(self, vmin, vmax, sin_out=False):
        super().__init__()
        self.register_buffer('vmin', torch.as_tensor(vmin), persistent=False)
        self.register_buffer('vmax', torch.as_tensor(vmax), persistent=False)
        self.sin_out = sin_out
        
    def forward(self, x):
        y = (x - self.vmin) / (self.vmax - self.vmin)
        
        if self.sin_out:
            return 2*y - 1
        return y
    
class InvRescaleTransform(torch.nn.Module):
    '''
    Inverse of :class:`RescaleTransform`.

    Example
    -------
        >>> rescale = InvRescaleTransform(vmin=-5, vmax=5)
        >>> inv_rescale = InvRescaleTransform(vmin=-5, vmax=5)
        >>> x = torch.linspace(-5, 5, 21)
        tensor([-5.0, -4.5, -4.0, -3.5, ..., 5.0])
        >>> inv_rescale(rescale(x))
        tensor([-5.0, -4.5, -4.0, -3.5, ..., 5.0])
    '''
    def __init__(self, vmin, vmax, sin_out=False):
        super().__init__()

        self.register_buffer('vmin', torch.as_tensor(vmin), persistent=False)
        self.register_buffer('vmax', torch.as_tensor(vmax), persistent=False)
        self.sin_out = sin_out
        
    def forward(self, x):
        if self.sin_out:
            x = (x+1) / 2
        
        y = x * (self.vmax - self.vmin) + self.vmin
        
        return y

class PseudoLogTransform(torch.nn.Module):
    '''
    Taking pseudo-logarithm `log10(x+eps)` and rescale the output to `[0,1]` or
    `[-1,1]`. Reimplantation of `xform_vis`.

    Arguments
    ---------
    vmax: float | torch.Tensor
        Upper bound of the input. The input range is `[0,max]`.

    eps : float | torch.Tensor
        A small number added to `log10(x+eps)` that allows `x=0`.

    sin_out: bool (optional)
        Rescale the output `[-1,1]`. Default: False, i.e. `[0,1]`.
    '''
    def __init__(self, vmax=1., eps=1e-7, sin_out=False):
        super().__init__()
        self.register_buffer('vmax', torch.as_tensor(vmax), persistent=False)
        self.register_buffer('eps', torch.as_tensor(eps), persistent=False)
        
        log_vmin = torch.log10(self.eps)
        log_vmax = torch.log10(self.vmax + self.eps)
        self._rescale = RescaleTransform(log_vmin, log_vmax, sin_out)
    
    def forward(self, x):
        y = torch.log10(x + self.eps)
        return self._rescale(y)

    @property
    def sin_out(self):
        return self._rescale.sin_out
    
class InvPseudoLogTransform(torch.nn.Module):
    '''
    Inverse of :class:`PseudoLogTransform`  Reimplantation of `inv_xform_vis`.
    '''
    def __init__(self, vmax=1., eps=1e-7, sin_out=False):
        super().__init__()
        self.register_buffer('vmax', torch.as_tensor(vmax), persistent=False)
        self.register_buffer('eps', torch.as_tensor(eps), persistent=False)
        
        log_vmin = torch.log10(self.eps)
        log_vmax = torch.log10(self.vmax + self.eps)
        self._inv_rescale = InvRescaleTransform(log_vmin, log_vmax, sin_out)
    
    def forward(self, x):
        y = self._inv_rescale(x)
        y = torch.pow(10., y) - self.eps
        return y

    @property
    def sin_out(self):
        return self._inv_rescale.sin_out

def transform_factory(cfg, device=None):
    '''
    Factory to create transformation and its inverse.

    Arguments
    ---------
    cfg : dict | None
        Configuration dictionary. If `None`, returns identity transformation.

        Type of tranformation are given by `cfg['method']`.
        `identity`   : no transformation
        `pseudo_log` : default, use :class:`PseudoLogTransform` and its inverse
        `rescale`    : use :class:`RescaleTransform` and its inverse

        The rest are keyward arguments to the class constructor.

    Returns
    -------
    xform, inv_xform: 
        Transformation and its inverse.

    Example
    -------

        Identity 
        >>> cfg = None 
        >>> cfg = dict(method='identity') ## also works
        >>> x = torch.linspace(0, 0.9, 101)
        >>> xform, inv_form = transform_factroy(cfg)
        >>> torch.allclose(x, xform(x))
        True
        >>> torch.allclose(x, inv_xform(x))
        True

        Pseudo-Log
        >>> cfg = dict(vmax=0.9, eps=1e-5)
        >>> cfg = dict(method='pseudo_log', vmax=0.9, eps=1e-5) ## same as above
        >>> x = torch.linspace(0, 0.9, 101)
        >>> xform, inv_xform = transform_factroy(cfg)
        >>> y = xform(x)
        >>> torch.allclose(x, inv_xform(y))
        True

        Rescale linearly
        >>> cfg = dict(method='rescale', vmin=0.1, vmax=0.9)
        >>> x = torch.linspace(0.1, 0.9, 101)
        >>> y = torch.linspace(0, 1, 101)
        >>> xform, inv_form = transform_factroy(cfg)
        >>> torch.allclose(y, xform(x))
        True
        >>> torch.allclose(x, inv_xform(y))
        True
    '''

    identity = torch.nn.Identity().to(device)
    if cfg is None:
        return identity, identity

    kwargs = cfg.copy()
    method = kwargs.pop('method', 'pseudo_log')

    if method == 'identity':
        return identity, identity

    if method == 'pseudo_log':
        xform  = PseudoLogTransform(**kwargs).to(device)
        inv_xform = InvPseudoLogTransform(**kwargs).to(device)
        return xform, inv_xform

    elif method == 'rescale':
        xform = RescaleTransform(**kwargs).to(device)
        inv_xform = InvRescaleTransform(**kwargs).to(device)
        return xform, inv_xform

    raise NotImplementedError(
        method, 'is not one of identity, pseudo_log, rescale'
    )
