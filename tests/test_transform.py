import random
import torch
import numpy as np
from inspect import signature

from tests.fixtures import torch_rng
from slar.transform import partial_xform_vis, xform_vis, inv_xform_vis
from slar.transform import (
    RescaleTransform, InvRescaleTransform,
    PseudoLogTransform, InvPseudoLogTransform
)

def is_monotonic(L):
    # not all increasing or not all decreasing
    return all(x>=y for x, y in zip(L, L[1:])) or \
           all(x<=y for x, y in zip(L, L[1:]))

def test_xform_vis(torch_rng):
    # test monotonicity and bounds
    x = torch.pow(10, torch.linspace(-7, 0, 10000))
    transformed = xform_vis(x)
    assert is_monotonic(transformed), "xform_vis is not monotonic"
    assert torch.all(transformed >= 0), "xform_vis is not bounded to 0"
    assert torch.all(transformed <= 1), "xform_vis is not bounded to 1"
    
    # test multi dim shape
    random_dim = torch.randint(1, 4, size=(1,), generator=torch_rng).item()
    random_size = torch.randint(1, 100, size=(random_dim,), generator=torch_rng).tolist()
    x_weird_shape = torch.pow(10, torch.rand(size=random_size)*(-10)+1e-10)
    assert xform_vis(x_weird_shape).shape == x_weird_shape.shape, "xform_vis does not preserve shape"
    
    # set eps = 1
    # expect [log(x+e) - log(e)] / [log(1+e) - log(e)] --> log(x+1) / log(2)
    transformed_eps1 = xform_vis(x, eps=1)
    expected = torch.log10(x+1)/np.log10(2)
    assert torch.allclose(transformed_eps1, expected), "xform_vis does not work as expected with eps=0"
    
    # test vmax
    vmax = torch.rand(1)*100
    assert torch.allclose(xform_vis(vmax, vmax=vmax), torch.tensor(1).float()), "xform_vis does not work as expected with vmax"
    
    # test sin_out
    assert torch.allclose(xform_vis(torch.tensor(0).float(), sin_out=True), torch.tensor(-1).float())
    assert torch.allclose(xform_vis(torch.tensor(1).float(), sin_out=True), torch.tensor(1).float())
    transformed_sinout = xform_vis(x, sin_out=True)
    assert torch.all(transformed_sinout >= -1), "xform_vis(sin_out=True) is not bounded to -1"
    assert torch.all(transformed_sinout <= 1), "xform_vis(sin_out=True) is not bounded to 1"

def test_inv_xform_vis(torch_rng):
    
    # test monotonicity and bounds
    x = torch.pow(10, torch.linspace(-5, 0, 500))
    transformed = xform_vis(x)
    assert is_monotonic(transformed), "inv_xform_vis is not monotonic"
    assert torch.allclose(x, inv_xform_vis(xform_vis(x))), "inv_xform_vis is not the inverse of xform_vis"
    assert torch.all(transformed >= 0), "xform_vis is not bounded to 0"
    assert torch.all(transformed <= 1), "xform_vis is not bounded to 1"

    
    # test multi dim shape
    random_dim = torch.randint(1, 4, size=(1,), generator=torch_rng).item()
    random_size = torch.randint(1, 100, size=(random_dim,), generator=torch_rng).tolist()
    x_weird_shape = torch.pow(10, torch.rand(size=random_size)*(-10)+1e-10)
    assert inv_xform_vis(x_weird_shape).shape == x_weird_shape.shape, "inv_xform_vis does not preserve shape"
    
    # set eps = 1
    # y = log(x+1) / log(2) --> x = 2^y - 1
    inv_transformed_eps1 = inv_xform_vis(x, eps=1)
    expected = torch.pow(2, x) - 1
    assert torch.allclose(expected, inv_transformed_eps1, atol=1e-6), "inv_xform_vis does not work as expected with eps=1"
    
    # test vmax
    vmax = torch.rand(1)*100
    assert torch.allclose(inv_xform_vis(torch.tensor(1).float(), vmax=vmax), vmax), "inv_xform_vis does not work as expected with vmax"
    
    # test sin_out
    assert torch.allclose(inv_xform_vis(torch.tensor(-1).float(), sin_out=True), torch.tensor(0).float())
    assert torch.allclose(inv_xform_vis(torch.tensor(1).float(), sin_out=True), torch.tensor(1).float())
    transformed_sinout = inv_xform_vis(x, sin_out=True)
    assert torch.all(transformed_sinout >= -1), "inv_xform_vis(sin_out=True) is not bounded to -1"
    assert torch.all(transformed_sinout <= 1), "inv_xform_vis(sin_out=True) is not bounded to 1"

def test_partial_xform_vis():
    d = {}
    xform, inv_xform = partial_xform_vis(d)
    
    assert signature(xform) == signature(xform_vis), "partial_xform_vis does not return the correct function"
    assert signature(inv_xform) == signature(inv_xform_vis), "partial_xform_vis does not return the correct function"
    
    d = {'eps': 1, 'vmax': 1, 'sin_out': True}
    xform, inv_xform = partial_xform_vis(d)
    for k,v in d.items():
        assert signature(xform).parameters[k].default == v, "partial_xform_vis does not return the correct function"
        assert signature(inv_xform).parameters[k].default == v, "partial_xform_vis does not return the correct function"
        
    xform, inv_xform = partial_xform_vis(None)
    
    x = torch.pow(10, torch.linspace(-7, 0, 10000))
    assert torch.allclose(xform(x), x), "xform from partial_xform_vis without kwargs isn't the identity"
    assert torch.allclose(inv_xform(x), x), "inv_xform from partial_xform_vis without kwargs isn't the identity"

def test_rescale():
    vmin = -5.
    vmax = 5.

    # without sin_out
    x = torch.linspace(vmin, vmax, 1001, dtype=float)
    y = torch.linspace(0, 1, 1001, dtype=float)

    rescale = RescaleTransform(vmin, vmax)
    inv_rescale = InvRescaleTransform(vmin, vmax)

    assert torch.allclose(y, rescale(x)), \
        'RescaleTransform fails to map [-5,5] -> [0,1]'

    assert torch.allclose(x, inv_rescale(y)), \
        'InvRescaleTransform fails to map [0,1] -> [-5,5]'


    # with sin_out
    rescale_sin = RescaleTransform(vmin, vmax, sin_out=True)
    inv_rescale_sin = InvRescaleTransform(vmin, vmax, sin_out=True)
    y_sin = torch.linspace(-1, 1, 1001, dtype=float)

    assert torch.allclose(y_sin, rescale_sin(x)), \
        'RescaleTransform (sin_out) fails to map [-5,5] -> [-1,1]'

    assert torch.allclose(x, inv_rescale_sin(y_sin)), \
        'InvRescaleTransform (sin_out) fails to map [-1,1] -> [-5,5]'

def test_pseudo_log():
    vmax = 0.5
    eps = 1e-5

    x = torch.linspace(0, vmax, 1001)
    y = torch.linspace(0, 1, 1001)

    log_trans = PseudoLogTransform(vmax, eps)
    inv_log_trans = InvPseudoLogTransform(vmax, eps)

    assert torch.allclose(log_trans(x), xform_vis(x,vmax,eps)), \
        'PseudoLogTransform != xform_vis'

    assert torch.allclose(inv_log_trans(y), inv_xform_vis(y,vmax,eps)), \
        'InvPseudoLogTransform != inv_xform_vis'
