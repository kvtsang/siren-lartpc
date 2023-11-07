import pytest
import torch
from slar.analysis import vis_bias


def test_vis_bias():
    target = torch.tensor([0.1, 0.2, 0.3, 0.4])
    pred = torch.tensor([0.2, 0.1, 0.3, 0.4])
    
    assert torch.allclose(vis_bias(target, pred), torch.tensor(1/3))
    
    threshold = 0.3
    assert torch.allclose(vis_bias(target, pred, threshold), torch.tensor(0.0))
    
    with pytest.raises(ValueError):
        vis_bias(target, pred[:-1])