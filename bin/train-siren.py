#!/usr/bin/env python3
import torch
import yaml
import fire
import os
from slar.train import train
from slar.utils import list_config, load_config

def main(config_file,device=None,lr=None,resume=None,max_epochs=None,max_iterations=None,ckpt_file=None,logdir=None):
    """
    An executable function for training Siren with a photon library
    
    Parameters
    ----------
    
    config_file : str
        Path to a yaml configuration file

    device : str
        cpu/gpu/mps (if software/hardware supported)

    lr : float
        Learning rate

    resume : bool
        If True, continue training from the last checkpoint (requires ckpt_file)

    max_epochs : int
        The maximum number of epochs before stop training

    max_iterations : int
        The maximum number of iterations before stop training

    ckpt_file : str
        Torch checkpoint file stored from training
    """
    cfg=dict()

    if not os.path.isfile(config_file) and config_file in list_config():
        cfg=load_config(config_file)
    else:
        with open(config_file,'r') as f:
            cfg=yaml.safe_load(f)

    cfg_update = dict()
    if max_epochs: cfg_update['max_epochs']=max_epochs
    if max_iterations: cfg_update['max_iterations']=max_iterations
    if resume: cfg_update['resume']=resume
    train_cfg = cfg.get('train',dict())
    train_cfg.update(cfg_update)
    cfg['train'] = train_cfg

    cfg_update = dict()
    if lr: cfg_update['lr']=lr
    optim_cfg = cfg.get('train').get('optimizer_param',dict())
    optim_cfg.update(cfg_update)
    cfg['train']['optimizer_param'] = optim_cfg

    
    cfg_update = dict()
    if ckpt_file: cfg_update['ckpt_file']=ckpt_file
    model_cfg = cfg.get('model',dict())
    model_cfg.update(cfg_update)
    cfg['model'] = model_cfg
    
    cfg_update = dict()
    if logdir: cfg_update['dir_name']=logdir
    logger_cfg = cfg.get('logger',dict())
    logger_cfg.update(cfg_update)
    cfg['logger'] = logger_cfg

    cfg_update = dict()
    if device: cfg_update['type']=device
    device_cfg = cfg.get('device',dict())
    device_cfg.update(cfg_update)
    cfg['device'] = device_cfg
    
    train(cfg)

    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    fire.Fire(main)
