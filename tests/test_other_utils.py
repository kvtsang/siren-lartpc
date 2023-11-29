import pytest
import torch
import slar.utils as other_utils
import os


def test_list_available_devices():
    devs = other_utils.list_available_devices()
    assert isinstance(devs, dict)
    assert 'cpu' in devs
    if torch.cuda.is_available():
        assert 'cuda' in devs
    if torch.backends.mps.is_available():
        assert 'mps' in devs


def test_get_device():
    devs = other_utils.list_available_devices()
    for dev in devs:
        assert other_utils.get_device(dev) == devs[dev]
    assert other_utils.get_device('invalid') is None


def test_import_from():
    assert other_utils.import_from('os.path') == os.path
    assert other_utils.import_from('torch.nn.functional') == torch.nn.functional


def test_get_config_dir():
    assert other_utils.get_config_dir().endswith('/config')


def test_list_config():
    configs = other_utils.list_config()
    assert isinstance(configs, list)
    assert len(configs) > 0
    assert isinstance(configs[0], str)
    full_paths = other_utils.list_config(full_path=True)
    assert isinstance(full_paths, list)
    assert len(full_paths) == len(configs)


def test_get_config():
    with pytest.raises(NotImplementedError):
        other_utils.get_config('invalid')
    configs = other_utils.list_config()
    for config in configs:
        assert other_utils.get_config(config) == os.path.join(other_utils.get_config_dir(), config + '.yaml')


def test_load_config():
    configs = other_utils.list_config()
    for config in configs:
        assert isinstance(other_utils.load_config(config), dict)
