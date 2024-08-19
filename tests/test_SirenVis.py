import os
from inspect import signature

import numpy as np
import pytest
import torch
import yaml
from slar.nets import SirenVis
from slar.transform import PseudoLogTransform, InvPseudoLogTransform
from slar.utils import to_plib

from photonlib import PhotonLib
from photonlib.meta import AABox
from tests.fixtures import fake_photon_library, writable_temp_file


@pytest.fixture
def cfg(fake_photon_library):
    config = f"""
    model:
      network:
        in_features: 3
        hidden_features: 4
        hidden_layers: 2
        out_features: 180
        first_omega_0: 30
        hidden_omega_0: 10
        outermost_linear: False
    photonlib:
      filepath: {fake_photon_library}
    transform_vis:
      eps: 1.0e-8
      sin_out: True
      vmax: 1.0
    """
    return yaml.safe_load(config)

@pytest.fixture
def slib(cfg):
    return SirenVis(cfg)

@pytest.fixture
def slib_hardsigmoid(cfg):
    _cfg = cfg.copy()
    _cfg['model']['hardsigmoid'] = True
    return SirenVis(_cfg)


def test_SirenVis_init(cfg, rng):
    net = SirenVis(cfg).to('cpu')
    
    out_features = cfg['model']['network']['out_features']
    assert net._n_outs == out_features, 'number of outputs is not as expected'
    assert isinstance(net.meta, AABox), 'meta is not AABox'
    assert net.device.type == 'cpu', 'device is not cpu'
    
    assert isinstance(net._xform_vis, PseudoLogTransform), \
        'xform_vis is not PseudoLogTransform'

    assert isinstance(net._inv_xform_vis, InvPseudoLogTransform), \
        'inv_xform_vis is not InvPseudoLogTransform'
    
    # test init_output_scale
    assert hasattr(net, 'output_scale'), 'output_scale is not an attribute of the network'
    assert isinstance(net.output_scale, torch.Tensor), 'output_scale is not a Tensor'
    assert torch.allclose(net.output_scale, torch.ones(out_features).float()), 'output_scale is not all ones'
    
    # test with str init
    random_scale = rng.uniform(0, 1, size=out_features)
    tempfile = writable_temp_file(suffix='.npy')
    np.save(tempfile, random_scale)
    _cfg = cfg.copy()
    _cfg['model']['output_scale'] = dict(init=tempfile)
    net = SirenVis(_cfg)
    assert torch.allclose(net.output_scale, torch.from_numpy(random_scale).float()), 'output_scale is not as expected'
    os.remove(tempfile)
    
    # test with array init
    _cfg = cfg.copy()
    _cfg['model']['output_scale'] = dict(init=random_scale.tolist())
    net = SirenVis(_cfg)
    assert torch.allclose(net.output_scale, torch.from_numpy(random_scale).float()), 'output_scale is not as expected'
    
    # test fix=False (i.e., let PMT scale array float)
    _cfg = cfg.copy()
    _cfg['model']['output_scale'] = dict(fix=False)
    net = SirenVis(_cfg)
    assert isinstance(net.output_scale, torch.nn.Parameter), 'output_scale is not a Parameter'
    assert torch.allclose(net.output_scale.data, torch.ones(out_features).float()), 'output_scale is not all ones'
    
    

def test_SirenVis_visibility(slib, slib_hardsigmoid, torch_rng):
    bounds = slib.meta.ranges
    
    random_pos = torch.rand(size=(10, 3), generator=torch_rng)*(bounds[:,1]-bounds[:,0]) + bounds[:,0]
    
    vis = slib.visibility(random_pos)
    assert vis.shape == (10, slib._n_outs), 'visibility shape is not as expected'
    assert torch.all(vis >= 0), 'visibility is not bounded by 0'
    assert torch.all(vis <= 1), 'visibility is not bounded by 1'
    
    vis_hardsig = slib_hardsigmoid.visibility(random_pos)
    assert slib_hardsigmoid._do_hardsigmoid is True, 'hardsigmoid is not enabled'
    assert torch.all(vis_hardsig != vis), 'visibility is the same as hardsigmoid'
    
    
def test_SirenVis_forward(slib, torch_rng):
    random_pos = 2*torch.rand(size=(10, 3), generator=torch_rng) - 1
    vis = slib(random_pos)
    assert vis.shape == (10, slib._n_outs), 'visibility shape is not as expected'
    assert torch.all(vis >= -1), 'visibility is not bounded by 0'
    assert torch.all(vis <= 1), 'visibility is not bounded by 1'
    
    
@pytest.mark.parametrize('do_hardsigmoid', [True, False])
@pytest.mark.parametrize('float_scale', [True, False])
def test_SirenVis_save_and_load(cfg, rng, do_hardsigmoid, float_scale):
    ckpt_file = writable_temp_file(suffix='.ckpt')
    out_features = cfg['model']['network']['out_features']
    random_scale = rng.uniform(0, 1, size=out_features)
    cfg['model']['output_scale'] = dict(init=random_scale.tolist(), fix=not float_scale)
    cfg['model']['hardsigmoid'] = do_hardsigmoid
    slib = SirenVis(cfg)
    slib.save_state(ckpt_file)
    
    # Test 2 loading methods: ckpt file and config
    slib2 = SirenVis.load(ckpt_file)

    _cfg = cfg.copy()
    _cfg['model']['output_scale'].pop('init')
    _cfg.pop('photonlib')
    _cfg.pop('transform_vis')
    _cfg['model']['ckpt_file'] = ckpt_file
    slib3 = SirenVis(_cfg)
    
    for i,loaded_slib in enumerate([slib2, slib3]):
        if float_scale:
            assert isinstance(loaded_slib.output_scale, torch.nn.Parameter), 'loaded output_scale is not a Parameter'
            assert torch.allclose(loaded_slib.output_scale.data, slib.output_scale.data), 'output_scale is not as expected'
        # FIXME - isn't output_scale always expected? -Kazu
        #else:
        #    assert torch.allclose(loaded_slib.output_scale, slib.output_scale), 'output_scale is not as expected'
        assert slib.config_xform == loaded_slib.config_xform, 'xform_cfg is not as expected'
        assert signature(slib._xform_vis) == signature(loaded_slib._xform_vis), 'xform_vis does not have expected signature'
        assert signature(slib._inv_xform_vis) == signature(loaded_slib._inv_xform_vis), 'inv_xform_vis does not have expected signature'
        assert torch.all(slib.meta.ranges == loaded_slib.meta.ranges), 'meta is not as expected'
        assert slib._do_hardsigmoid == loaded_slib._do_hardsigmoid == do_hardsigmoid, 'hardsigmoid is not as expected'

def test_SirenVis_to_plib(slib, fake_photon_library, rng):
    _plib = PhotonLib.load(fake_photon_library)
    
    # no batch size
    plib = to_plib(slib,_plib.meta)
    assert plib.vis.shape == _plib.vis.shape, 'plib vis shape is not as expected'
    
    # batch size
    random_batch_size = rng.integers(1, 100)
    plib_batched = to_plib(slib, _plib.meta, batch_size=random_batch_size)
    assert plib_batched.vis.shape == _plib.vis.shape, 'plib vis shape is not as expected'
    assert torch.allclose(plib_batched.vis,plib.vis, rtol=1e-4), 'batched plib != non-batched plib'
