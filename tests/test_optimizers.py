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
    cfg = {
        'train': {
            'optimizer_class': 'Adam', 
            'optimizer_param': {'lr': lr},
            'scheduler_class': 'StepLR',
            'scheduler_param': {'step_size': rng.integers(1, 10)},
        },
        'model': {'ckpt_file': filename}
    }
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
    # test optimizer only
    # remove scheduler from factory_cfg
    cfg = factory_cfg.copy()
    cfg['train'].pop('scheduler_class')
    cfg['train'].pop('scheduler_param')

    params = [torch.randn(3, 5)]
    opt, sch, epoch = optimizer_factory(params, factory_cfg)
    assert isinstance(opt, torch.optim.Adam)
    assert sch == None
    assert epoch == 0
    assert get_lr(opt) == factory_cfg['train']['optimizer_param']['lr']

    opt = torch.optim.Adam([torch.randn(3, 5)])
    true_epoch = rng.integers(0, 5000)
    state = {'optimizer': opt.state_dict(), 'epoch': true_epoch}

    torch.save(state, factory_cfg['model']['ckpt_file'])
    _cfg = cfg.copy()
    _cfg['train']['resume'] = True
    opt, sch, epoch = optimizer_factory(params, _cfg)
    assert isinstance(opt, torch.optim.Adam)
    assert sch == None
    assert get_lr(opt) == _cfg['train']['optimizer_param']['lr']
    assert epoch == true_epoch

def test_scheduler_factory(factory_cfg, rng):
    params = [torch.randn(3, 5)]
    opt, sch, epoch = optimizer_factory(params, factory_cfg)
    assert isinstance(opt, torch.optim.Adam)
    assert isinstance(sch, torch.optim.lr_scheduler.StepLR)
    assert epoch == 0
    assert get_lr(opt) == factory_cfg['train']['optimizer_param']['lr']

    step_size = factory_cfg['train']['scheduler_param']['step_size']
    for last_epoch in range(step_size+3):
        opt.step()
        sch.step()
    last_lr = get_lr(opt)

    state = {
        'optimizer': opt.state_dict(), 
        'scheduler': sch.state_dict(), 
        'epoch': last_epoch,
    }

    torch.save(state, factory_cfg['model']['ckpt_file'])
    _cfg = factory_cfg.copy()
    _cfg['train']['resume'] = True

    # drop lr from cfg; restore lr from checkpoint.
    _cfg['train']['optimizer_param'].pop('lr')

    opt, sch, epoch = optimizer_factory(params, _cfg)
    assert isinstance(opt, torch.optim.Adam)
    assert isinstance(sch, torch.optim.lr_scheduler.StepLR)
    assert epoch == last_epoch
    assert get_lr(opt) == last_lr
