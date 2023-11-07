
from inspect import signature
import yaml
import torch
import pytest
from slar.io import PhotonLibDataset
from photonlib import PhotonLib
from slar.transform import xform_vis, inv_xform_vis

from torch.utils.data import Dataset

from tests.fixtures import rng, fake_photon_library

@pytest.fixture
def cfg_filled(fake_photon_library):
    config = f"""
    data:
      dataset:
        weight:
          factor: 1.0e+8
          method: vis
          theshold: 1.0e-6
    photonlib:
      filepath: {fake_photon_library} 
    transform_vis:
      eps: 1.0e-8
      sin_out: True
      vmax: 0.9
    """
    return yaml.safe_load(config)

@pytest.fixture
def cfg_empty(fake_photon_library):
    config = f"""
    data:
      dataset:
        None: None
    photonlib:
      filepath: {fake_photon_library} 
    """

    return yaml.safe_load(config)


def test_PhotonLibDataset_constructor(cfg_filled, cfg_empty):
    ds_filled = PhotonLibDataset(cfg_filled)
    ds_unfilled = ds = PhotonLibDataset(cfg_empty)
    assert isinstance(ds, Dataset)
    
    # test visibility transforms are as expected
    for k,v in cfg_filled['transform_vis'].items():
        assert signature(ds_filled.xform_vis).parameters[k].default == v, "Dataset does not assign parameters from config to xform_vis"
        assert signature(ds_filled.inv_xform_vis).parameters[k].default == v, "Dataset does not assign parameters from config to inv_xform_vis"

    # make sure visibilities are as expected
    assert torch.allclose(ds.visibilities, ds.xform_vis(ds.plib.vis)), "Dataset visibilities haven't been transformed!"
    # make sure positions are normalized
    assert torch.any(ds.positions < 0) and torch.any(ds.positions > 0), 'expected positions on both sides of x=0'
    assert torch.all(ds.positions >= -1), 'expected positions to be bounded by -1'
    assert torch.all(ds.positions <= 1), 'expected positions to be bounded by 1'
    
    # make sure weights are as expected 
    assert hasattr(ds_filled, 'weights'), "Dataset doesn't have weights attribute"
    assert hasattr(ds_unfilled, 'weights'), "Dataset doesn't have weights attribute"
    assert ds_filled.weights.shape == ds.visibilities.shape, "Dataset weights shape doesn't match visibilities shape"
    assert torch.all(ds_filled.weights >= cfg_filled['data']['dataset']['weight']['theshold']), "Dataset weights are not bounded by threshold"
    
def test_PhotonLib_len(cfg_filled):
    ds = PhotonLibDataset(cfg_filled)
    assert len(ds) == ds.plib.meta.shape.prod().item(), "Dataset length isn't equal to number of voxels in photonlib"
    
    
def test_PhotonLib_getitem(cfg_filled, cfg_empty):
    datasets = [PhotonLibDataset(cfg_filled), PhotonLibDataset(cfg_empty)]
    
    for ds in datasets:
        # test __getitem__
        idx = torch.randint(0, len(ds), size=(1,)).item()
        item = ds[idx]
        assert isinstance(item, dict), "Dataset __getitem__ doesn't return a dict"
        for key in ['position', 'value', 'weight']:
            assert key in item, f"Dataset __getitem__ doesn't return a dict with key '{key}'"
            
        # test slicing
        idxs = torch.randint(0, len(ds), size=(10,)).tolist()
        ds[idxs]