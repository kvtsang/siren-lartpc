from inspect import signature
import yaml
import torch
import pytest
from slar.io import PLibDataLoader
from slar.transform import xform_vis, inv_xform_vis

from tests.fixtures import rng, fake_photon_library

@pytest.fixture
def cfg_default(fake_photon_library):
    cfg = yaml.safe_load(f'''
    data:
      dataset:
        weight:
          factor: 1.0e+8
          method: vis
          theshold: 1.0e-6
      loader:
        batch_size: 2
        shuffle: true
    photonlib:
      filepath: {fake_photon_library} 
    transform_vis:
      eps: 1.0e-8
      sin_out: True
      vmax: 0.9

    ''')
    return cfg

@pytest.fixture
def cfg_empty(fake_photon_library):
    cfg = yaml.safe_load(f'''
    photonlib:
       filepath: {fake_photon_library} 
    ''')

    return cfg

def test_PLibDataLoader_ctor(cfg_default, cfg_empty):
    ds = PLibDataLoader(cfg_default)
    ds_empty = PLibDataLoader(cfg_empty)

    for k,v in cfg_default['transform_vis'].items():
        assert signature(ds.xform_vis).parameters[k].default==v, \
            'xform_vis parameter(s) not assigned correctly'
        assert signature(ds.inv_xform_vis).parameters[k].default==v, \
            'inv_xform_vis parameter(s) not assigned correctly'

    assert ds.get_weight==ds.get_weight_by_vis,  'incorrect weighing scheme'

    assert ds._batch_mode, 'dataloader should be in batch mode'
    assert not ds_empty._batch_mode, 'dataloader should not be in batch mode'


def test_PLibDataLoader_getitem(cfg_default, cfg_empty):
    ds_empty = PLibDataLoader(cfg_empty)

    n_batch = 0
    for batch in ds_empty:
        n_batch += 1

    plib = ds_empty._plib
    assert n_batch==1, f'dataloader should only have 1 batch ({n_batch})'
    assert torch.allclose(batch['value'], plib.vis), f'incorrect vis values'
    assert batch['weight']==1, 'weight != 1'
    
    pos = batch['position']
    assert torch.all((pos >=  -1) & (pos <= 1)), f'coordinates not normalized'

    # ---------------------------------
    ds = PLibDataLoader(cfg_default)

    n_batch = 0
    n_pts = 0
    for batch in ds:
        n_batch += 1
        n_pts += len(batch['value'])

    plib = ds_empty._plib
    assert n_batch==len(ds), f'number of batches does not match'
    assert n_pts==len(plib), f'nubmer of voxels does not match'
