import torch

def get_lr(opt):
    '''
    Read the learning rate from torch.optim.Optimizer

    Parameters
    ----------
    opt : torch.optim.Optimizer
        The subject optimizer instance.

    Returns
    -------
    float
        The current learning rate value.
    '''
    for ps in opt.param_groups:
        return ps['lr']

def load_optimizer_state(filename, opt, sch=None):
    '''
    Function to read the optimizer parameters from a checkpoint file.

    Parameters
    ----------
    filename : str
        The checkpoint file path+name to extract the optimizer state.
    opt : torch.optim.Optimizer
        The optimizer instance to load the parameters for.
    sch : torch.optim.lr_scheduler, optional
        Learning rate scheduler

    Returns
    -------
    float
        The training epoch spent before the optimizer state was saved.
    '''
    epoch = 0
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')
        
        opt.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint.keys():
            epoch = int(checkpoint['epoch'])

        if sch is not None:
            sch.load_state_dict(checkpoint['scheduler'])
    return epoch


def optimizer_factory(params, cfg):
    '''
    Function to crate an optimizer for training siren

    Parameters
    ----------
    params : dict
        The parameters of an optimizer instance constructor.
    cfg : dict
        The configuration parameters for this factory method. The 'optimizer_class'
        is the string name of an optimzier class. If 'resume' is True, 'ckpt_file'
        will be looked for loading the optimizer's state from the previous training.

    Returns
    -------
    opt: torch.optim.Optimizer
        An instance of an optimizer with the parameters eitehr configured
        by the input or loaded from a checkpoint file.

    sch: torch.optim.lr_scheduler
        An instance of learning rate scheduler, if 'scheduler_class' is given in the cfg.
        Otherwise returns None

    epoch: float
        Current epoch. Restored from checkpoint for `resume=True`. 
    '''
    if 'optimizer_class' in cfg['train']:
        opt_class = cfg['train']['optimizer_class']
        opt_param = cfg['train']['optimizer_param']
    else:
        opt_class = cfg['train']['optimizer']['name']
        opt_param = cfg['train']['optimizer']['parameters']

    if not hasattr(torch.optim, opt_class):
        raise RuntimeError(f'torch.optim has no optimizer called {opt_class}')

    opt = getattr(torch.optim, opt_class)(params,**opt_param)
    print('[opt_factory] optimizer',opt_class)
    print('[opt_factory] parameters',opt_param)

    sch = scheduler_factory(opt, cfg)

    epoch = 0
    if cfg['train'].get('resume'):
        ckpt_file = cfg['model'].get('ckpt_file')
        if not ckpt_file:
            raise RuntimeError('cannot "resume" without the model checkpoint file')
        print(f'[opt_factory] loading the optimizer state from {ckpt_file}')
        epoch = load_optimizer_state(ckpt_file, opt, sch)

        if opt_param.get('lr', None) is not None:
            for p in opt.param_groups:
                p['lr'] = opt_param['lr']
        print(f'[opt_factory] current lr = {get_lr(opt)}')

    return opt, sch, epoch

def scheduler_factory(opt, cfg):
    '''
    Factory to create learning rate scheduler.

    Parameters
    ----------
    opt: torch.optim.Optimizer
        An instance of optimizer.
    cfg: dict
        Confguration of a lr_scheduler.
        'scheduler_class' - class name string in torch.optim.lr_scheduler
        'scheduler_param' - keyword argumets passed to the scheduler contructor

    Returns
    -------
    sch: torch.optim.lr_scheduler or None
        An instance of lr scheduler. If 'scheduler_class' is not missing,
        returns None.
    '''
    train_cfg = cfg.get('train', {})

    sch_class = train_cfg.get('scheduler_class')
    sch_param = train_cfg.get('scheduler_param', {})

    if sch_class is None:
        return None

    if not hasattr(torch.optim.lr_scheduler, sch_class):
        raise RuntimeError(
            f'torch.optim.lr_scheduler has no scheduler called {sch_class}'
        )

    Scheduler = getattr(torch.optim.lr_scheduler, sch_class)
    sch = Scheduler(opt, **sch_param)
    
    print('[opt_factory] scheduler', sch_class)
    print('[opt_factory] parameters', sch_param)

    return sch
