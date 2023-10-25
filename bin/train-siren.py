#!/usr/bin/env python3
import torch
import yaml
import fire
from slar.train import train

def main(config_file,device=None,lr=None,resume=None,max_epochs=None,max_iterations=None,ckpt_file=None,logdir=None):
    """
    An executable function for training Siren with a photon library
    
    Args:
        config_file (str): path to a yaml configuration file

        device (str): cpu/gpu/mps (if software/hardware supported)

        lr (float): learning rate

        resume (bool): if True, continue training from the last checkpoint (requires ckpt_file)

        max_epochs (int): the maximum number of epochs before stop training

        max_iterations (int): the maximum number of iterations before stop training

        ckpt_file (str): torch checkpoint file stored from training
    """
    cfg=dict()
    with open(config_file,'r') as f:
        cfg=yaml.safe_load(f)

    cfg_update = dict()
    if lr: cfg_update['lr']=lr
    if max_epochs: cfg_update['max_epochs']=max_epochs
    if max_iterations: cfg_update['max_iterations']=max_iterations
    if resume: cfg_update['resume']=resume
    train_cfg = cfg.get('train',dict())
    train_cfg.update(cfg_update)
    cfg['train'] = train_cfg
    
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
