import torch

def load_optimizer_state(filename,opt):
    epoch = 0
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')
        
        opt.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint.keys():
            epoch = int(checkpoint['epoch'])
    return epoch


def optimizer_factory(params,cfg):

    opt_class = cfg['train']['optimizer_class']
    opt_param = cfg['train']['optimizer_param']

    if not hasattr(torch.optim, opt_class):
        raise RuntimeError(f'torch.optim has no optimizer called {opt_class}')

    opt = getattr(torch.optim, opt_class)(params,**opt_param)
    print('[opt_factory] optimizer',opt_class)
    print('[opt_factory] parameters',opt_param)

    epoch = 0
    if cfg['train'].get('resume'):
        ckpt_file = cfg['model'].get('ckpt_file')
        if not ckpt_file:
            raise RuntimeError('cannot "resume" without the model checkpoint file')
        print(f'[opt_factory] loading the optimizer state from {ckpt_file}')
        epoch = load_optimizer_state(ckpt_file,opt)

    return opt, epoch

