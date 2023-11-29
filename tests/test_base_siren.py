from slar.base import Siren, SineLayer

import numpy as np
import torch
import pytest
from tests.fixtures import rng, num_pmt


def test_Siren_init(rng):
    in_features = 3
    hidden_features = 4
    hidden_layers = 20
    out_features = 180
    first_omega_0 = 30
    hidden_omega_0 = 10
    outermost_linear = False
    
    siren = Siren(in_features, hidden_features, hidden_layers, out_features, \
                  outermost_linear, first_omega_0, hidden_omega_0)
    
    assert len(siren.net) == hidden_layers + 2, 'number of layers is not as expected'
    assert all(isinstance(net, SineLayer) for net in siren.net), 'some layers not SineLayer'
    assert siren.net[0].is_first, 'first layer is not marked as first'
    assert all(not net.is_first for net in siren.net[1:]), 'non-first layer is marked as first'
    assert siren.net[0].omega_0 == first_omega_0, 'first layer omega_0 is not as expected'
    assert all(net.omega_0 == hidden_omega_0 for net in siren.net[1:]), 'last layer omega_0 is not as expected'
    assert siren.net[0].in_features == in_features, 'first layer in_features is not as expected'
    
    # check that weights are initialized as expected
    assert torch.all(siren.net[0].linear.weight >= -1 / in_features), 'first layer weight lower bound is not as expected'
    assert torch.all(siren.net[0].linear.weight <= 1 / in_features), 'first layer weight upper bound is not as expected'
    for net in siren.net[1:]:
        assert torch.all(net.linear.weight >= -np.sqrt(6 / in_features) / hidden_omega_0), 'sine layer weight lower bound is not as expected'
        assert torch.all(net.linear.weight <= +np.sqrt(6 / in_features) / hidden_omega_0), 'sine layer weight upper bound is not as expected'
    
    outermost_linear = True
    siren = Siren(in_features, hidden_features, hidden_layers, out_features, \
                  outermost_linear, first_omega_0, hidden_omega_0)
    assert isinstance(siren.net[-1], torch.nn.Linear), 'last layer is not Linear'
    assert torch.all(siren.net[-1].weight >= -1 / in_features), 'last layer weight lower bound is not as expected'
    assert torch.all(siren.net[-1].weight <= 1 / in_features), 'last layer weight upper bound is not as expected'
    
    
def test_Siren_forward(rng):
    in_features = 3
    hidden_features = 4
    hidden_layers = 2
    out_features = 180
    first_omega_0 = 30
    hidden_omega_0 = 10
    outermost_linear = False
    
    siren = Siren(in_features, hidden_features, hidden_layers, out_features, \
                  outermost_linear, first_omega_0, hidden_omega_0)
    
    x = torch.rand(1, in_features)
    y = siren(x)
    assert y.shape == (1, out_features), 'output shape is not as expected'
    assert torch.all(torch.abs(y) >= -1), 'output is not bounded by [-1, 1]'
    assert torch.all(torch.abs(y) <= 1), 'output is not bounded by [-1, 1]'
    
    x = torch.rand(10, in_features)
    y = siren(x)
    assert y.shape == (10, out_features), 'output shape is not as expected'
    assert torch.all(torch.abs(y) >= -1), 'output is not bounded by [-1, 1]'
    assert torch.all(torch.abs(y) <= 1), 'output is not bounded by [-1, 1]'
    
    
def test_Siren_forward_with_activations(rng):
    in_features = 3
    hidden_features = 4
    hidden_layers = 53
    out_features = 180
    first_omega_0 = 30
    hidden_omega_0 = 10
    outermost_linear = True
    
    siren = Siren(in_features, hidden_features, hidden_layers, out_features, \
                  outermost_linear, first_omega_0, hidden_omega_0)
    
    x = torch.rand(1, in_features)
    activations = siren.forward_with_activations(x)
    assert len(activations) == 2*(hidden_layers+2), 'number of activations is not as expected'

    x = torch.rand(1, in_features, requires_grad=True)
    activations = siren.forward_with_activations(x, retain_grad=True)
    assert all(activation.requires_grad for activation in activations.values()), 'activations do not require grad'
    
    