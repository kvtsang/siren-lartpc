from slar.utils import CSVLogger
import os
import pytest
from inspect import signature
from slar.analysis import vis_bias
import torch
from functools import partial
from tests.fixtures import writable_temp_file

@pytest.fixture
def cfg():
    tmpdir = writable_temp_file(suffix='.csv')
    directory, filename = os.path.split(tmpdir)
    cfg = {
        'logger': {
            'dir_name': directory,
            'file_name': filename,
            'log_every_nsteps': 2,
            'analysis': {
                'vis_bias': {'threshold': 1e-6}
            }
        }
    }
    return cfg

@pytest.fixture
def logger(cfg):
    logger = CSVLogger(cfg)
    yield logger
    if os.path.exists(logger.logfile):
        os.remove(logger.logfile)
    os.rmdir(logger.logdir)

def test_CSVLogger_analysis(logger):
    assert len(logger._analysis_dict) == 1
    assert signature(logger._analysis_dict['vis_bias']) == signature(partial(vis_bias, threshold=1e-6))
    

def test_CSVLogger_logdir(logger):
    assert os.path.basename(logger.logdir) == 'version-00'
    
    
def test_CSVLogger_record(logger):
    keys = ['key1', 'key2']
    vals = [1, 2]
    logger.record(keys, vals)
    assert logger._dict == {'key1': 1, 'key2': 2}

def test_CSVLogger_step(logger):
    label = torch.as_tensor([1, 2, 3])
    pred = torch.as_tensor([1, 2, 3])
    pred_flipped = torch.flip(pred, dims=(0,))
    logger.step(0, label, pred)
    assert logger._dict == {'vis_bias': 0.0}
    
    # no change as log_every_nsteps = 2
    logger.step(1, label, pred_flipped)
    assert logger._dict == {'vis_bias': 0.0}

    # change as log_every_nsteps = 2
    logger.step(2, label, pred_flipped)
    assert logger._dict == {'vis_bias': torch.tensor(2/3)}


def test_CSVLogger_write(logger):
    keys = ['key1', 'key2']
    vals = [1, 2]
    logger.record(keys, vals)
    logger.write()
    with open(logger.logfile, 'r') as f:
        assert f.read() == 'key1,key2\n1.000000,2.000000\n'

def test_CSVLogger_close(logger):
    assert logger._fout is None
    logger.close() # does nothing
    keys = ['key1', 'key2']
    vals = [1, 2]
    logger.record(keys, vals)
    logger.write()
        
    assert logger._fout is not None
    logger.close()
    assert logger._fout.closed

def test_CSVLogger_flush(logger):
    assert logger._fout is None
    logger.flush() # does nothing
    keys = ['key1', 'key2']
    vals = [1, 2]
    logger.record(keys, vals)
    logger.write()
        
    assert logger._fout is not None
    logger.flush()
    
def test_multiple_CSVLogger_dirs():
    tmpdir = writable_temp_file(suffix='.csv')
    directory, filename = os.path.split(tmpdir)
    cfg = {
        'logger': {
            'dir_name': directory,
            'file_name': filename,
            'log_every_nsteps': 2,
            'analysis': {
                'vis_bias': {'threshold': 1e-6}
            }
        }
    }
    logger1 = CSVLogger(cfg)
    logger2 = CSVLogger(cfg)
    
    assert logger1.logdir != logger2.logdir
    assert os.path.basename(logger1.logdir) == 'version-00'
    assert os.path.basename(logger2.logdir) == 'version-01'
    os.rmdir(logger1.logdir)
    os.rmdir(logger2.logdir)