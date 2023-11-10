import torch
import os
from slar.optimizers import get_lr, load_optimizer_state, optimizer_factory
from tests.fixtures import rng, writable_temp_file
from inspect import signature
import pytest

@pytest.fixture
def factory_cfg(rng):
    filename = writable_temp_file(suffix='.ckpt')
    lr = 10**rng.uniform(-6, -2)
    cfg = {'train': {'optimizer_class': 'Adam', 
                    'optimizer_param': {'lr': lr}}, 
            'model': {'ckpt_file': filename}}
    yield cfg
    os.remove(filename)

def test_get_lr(rng):
    opt = torch.optim.Adam([torch.randn(3, 5)])
    assert get_lr(opt) == signature(torch.optim.Adam).parameters['lr'].default
    
    lr = 10**rng.uniform(-6, -2)
    opt = torch.optim.Adam([torch.randn(3, 5)], lr=lr)
    assert get_lr(opt) == lr

def test_load_optimizer_state(rng):
    opt = torch.optim.Adam([torch.randn(3, 5)])
    epoch = rng.integers(0, 5000)
    state = {'optimizer': opt.state_dict(), 'epoch': epoch}
    filename = writable_temp_file(suffix='.ckpt')
    torch.save(state, filename)
    epoch_loaded = load_optimizer_state(filename, opt)
    assert epoch_loaded == epoch

def test_optimizer_factory(factory_cfg, rng):
    params = [torch.randn(3, 5)]
    opt, epoch = optimizer_factory(params, factory_cfg)
    assert isinstance(opt, torch.optim.Adam)
    assert epoch == 0
    assert get_lr(opt) == factory_cfg['train']['optimizer_param']['lr']

    opt = torch.optim.Adam([torch.randn(3, 5)])
    true_epoch = rng.integers(0, 5000)
    state = {'optimizer': opt.state_dict(), 'epoch': true_epoch}

    torch.save(state, factory_cfg['model']['ckpt_file'])
    _cfg = factory_cfg.copy()
    _cfg['train']['resume'] = True
    opt, epoch = optimizer_factory(params, _cfg)
    assert isinstance(opt, torch.optim.Adam)
    assert get_lr(opt) == _cfg['train']['optimizer_param']['lr']
    assert epoch == true_epoch