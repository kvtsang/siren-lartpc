from __future__ import annotations

import torch
import numpy as np
import tqdm

from slar.base import Siren
from slar.transform import partial_xform_vis
from photonlib import AABox
class WeightedL2Loss(torch.nn.Module):
    '''
    A simple loss module that implements a weighted MSE loss
    '''

    def __init__(self,reduce_method=torch.mean):
        super().__init__()
        self.reduce = reduce_method
    
    def forward(self,pred, target, weight=1.):
        loss = weight * (pred - target)**2
        return self.reduce(loss)

class SirenVis(Siren):
    '''
    Siren class implementation for LArTPC optical photon transport
    '''
    
    def __init__(self, cfg : dict, meta = None):
        '''
        Constructor takes two different modes depending on the arguments.

        Mode 1 : built entirely from the configuration file
            
            This mode is taken if both ckpt_file and cfg['model']['ckpt_file'] are None
            
            The model constructs all attributes based on the configuration file. In particular,
            this includes the function input coordinate information (self._meta), visibility
            forward and inverse transformation functions (self._xform_vis, self._inv_xform_vis),
            a flag to apply sigmoid on the output of Siren (self._do_hardsigmoid), and the scaling
            factor applied to each visibility (self.output_scale).

        Mode 2: load everything but the network architecture from the configuration file
            
            This mode is enabled if cfg['model']['ckpt_file'] is provided.
            
            Other than the model architecture, all attributes are loaded from the ckpt_file.
            This is to ensure the configuration used for training, and hence for the model stored
            in the ckpt_file, to remain the same after loading the state. 

        Parameters
        ----------
        cfg : dict
            Configuration parameters
        ckpt_file : str or dict
            Checkpoint file path in string. If provided, ignores some parameters in cfg. See the explanation above.
        '''
        self.config_model = cfg['model']
        
        # initialize Siren class
        super().__init__(**self.config_model['network'])

        self._n_outs = self.config_model['network']['out_features']

        if self.config_model.get('ckpt_file'):
            filepath=self.config_model.get('ckpt_file')
            print('[SirenVis] creating from checkpoint',filepath)
            with open(filepath,'rb') as f:
                model_dict = torch.load(f, map_location='cpu')
                self.load_model_dict(model_dict)
            return

        # create meta
        if meta is not None:
            self._meta = meta
        elif 'photonlib' in cfg:
            self._meta = AABox.load(cfg['photonlib']['filepath'])

        # transform functions
        self.config_xform = cfg.get('transform_vis')
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self.config_xform)
        
        # extensions for visiblity model
        self._init_output_scale(self.config_model)
        self._do_hardsigmoid = self.config_model.get('hardsigmoid', False)


    @property
    def meta(self):
        '''
        Access photonlib.meta.AABox that stores the volume definition (xyz range) for this model.

        Returns
        -------
        AABox
            The volume definition for this model.
        '''
        return self._meta


    @property
    def device(self):
        '''
        Access the device this model is on. This function assumes all parameters are on the same device.

        Returns
        -------
        torch.device
            The device ID where this model instance resides on
        '''
        return next(self.parameters()).device


    @property
    def n_pmts(self):
        return self._n_outs    


    def update_meta(self, ranges:torch.Tensor):
        self._meta.update(ranges)

    def visibility(self, x):
        '''
        A function meant for analysis/inference (not for training) that returns
        the visibilities for all PMTs given the position(s) in x. Note x is not 
        a normalized coordinate.

        Parameters
        ----------
        x : torch.Tensor
            A (or an array of) 3D point in the absolute coordinate

        Returns
        -------
        torch.Tensor
            Holds the visibilities in linear scale for the position(s) x.
        '''
        device = x.device 
        pos = x
        squeeze=False
        if len(x.shape) == 1:
            pos = pos[None,:]
            squeeze=True
        vis = torch.zeros(size=(pos.shape[0],self.n_pmts),dtype=torch.float32).to(self.device)
        mask = self.meta.contain(pos)
        vis[mask] = self(self.meta.norm_coord(pos[mask]).to(self.device)).to(device)
        vis[mask] = self._inv_xform_vis(vis[mask])
        return vis if not squeeze else vis.squeeze()


    def forward(self, x):
        '''
        A function meant for training. The input position(s) x is assumed to be normalized
        in the range of -1 to 1 along each axis. The normalization is done within the 
        dataset class (see io.PhotonLibDataset).

        Parameters
        ----------
        x : torch.Tensor
            A (or an array of) 3D point in the normalized coordinate

        Returns
        -------
        torch.Tensor
            Holds the visibilities in log-scale for the position(s) x.
        '''
        assert not torch.any(torch.lt(x,-1) | torch.gt(x,1)), f"The input contains a value out of range [-1,1]"

        out = super().forward(x)

        if self._do_hardsigmoid:
            out =  torch.nn.functional.hardsigmoid(out)
            
        out = out * self.output_scale
        
        return out


    def model_dict(self, opt=None, epoch=-1):
        model_dict=dict(state_dict  = self.state_dict(),
                        xform_cfg   = self.config_xform,
                        model_cfg   = self.config_model,
                        aabox_ranges= self._meta.ranges,
                       )
        # check if output_scale should be saved
        #pnames = [ name for name, p in self.named_parameters()]
        #if not 'output_scale' in pnames:
        #    state_dict['output_scale'] = self.output_scale 
        if opt:
            model_dict['optimizer'] = opt.state_dict()
        if epoch>=0:
            model_dict['epoch'] = epoch
        return model_dict

    def save_state(self, filename, opt=None, epoch=-1):
        '''
        Stores the network model and optimizer (and some hyper-) parameters to a binary file.

gpu
        Parameters
        ----------        
        filename : str
            The name of checkpoint file to be stored.
        opt : torch.optim.Optimizer
            The optimizer instance to store the optimizer state in the same file.
        epoch : float
            The epoch count of training.
        '''
        
        print('[SirenVis] saving the state ',filename)
        torch.save(self.model_dict(opt,epoch),filename)


    def load_model_dict(self, model_dict):
        '''
        Loads the network model and optimizer (and some hyper-) parameters from a dictionary
        Parameters
        ----------
        model_dict : dict
            Contains all model parameters necessary to re-instantiate the mode at the checkpoint.

        '''
        self.config_model = model_dict.get('model_cfg')
        self.config_xform = model_dict.get('xform_cfg')
        if self.config_model is None:
            raise KeyError('The model dictionary is lacking the "model_cfg" data')
        
        # 2024-03-14 xform_cfg can be None        
        #if self.config_xform is None:
        #    raise KeyError('The model dictionary is lacking the "xform_cfg" data')

        # 2024-03-11 Kazu - for backward compatibility
        #if 'do_hardsigmoid' in model_dict:
        #    self.config_model['do_hardsigmoid'] = model_dict['do_hardsigmoid']
        #if not 'output_scale' in self.config_model:
        #    self.config_model['output_scale']=dict(fix=True)

        self._init_output_scale(self.config_model)
        self._do_hardsigmoid = self.config_model.get('hardsigmoid', False)
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self.config_xform)

        from photonlib import AABox
        self._meta = AABox(model_dict['aabox_ranges'])

        # 2024-03-04 Kazu - For backward compatibility
        state_dict = model_dict['state_dict']
        if 'input_scale' in state_dict:
            state_dict.pop('input_scale')
        if 'scale' in model_dict.keys():
            state_dict['output_scale'] = model_dict['scale']
        #if not hasattr(self,'output_scale'):
        #    self.output_scale = state_dict.pop('output_scale')

        self.load_state_dict(model_dict['state_dict'])

    @classmethod
    def load(cls, cfg_or_fname: str | dict ):
        '''
        Constructor method that can take either a config dictionary or the data file path

        Parameters
        ----------
        cfg_or_fname : str
            If string type, it is interpreted as a path to a photon library data file.
            If dictionary type, it is interpreted as a configuration.
        '''

        if isinstance(cfg_or_fname,dict):
            if not 'model' in cfg_or_fname:
                raise KeyError('The configuration dictionary must contain model')
            if 'ckpt_file' in cfg_or_fname['model']:
                filepath=cfg_or_fname['model']['ckpt_file']
            else:
                return cls(cfg_or_fname)
        elif isinstance(cfg_or_fname,str):
            filepath=cfg_or_fname
        else:
            raise ValueError(f'The argument of load function must be str or dict (received {cfg_or_fname} {type(cfg_or_fname)})')

        print('[SirenVis] creating from checkpoint',filepath)
        with open(filepath,'rb') as f:

            model_dict = torch.load(f, map_location='cpu')

            return cls.create_from_model_dict(model_dict)


    @classmethod
    def create_from_model_dict(cls, model_dict):

        cfg = dict(model=model_dict['model_cfg'], transform_vis=model_dict['xform_cfg'])

        net = cls(cfg)

        net.load_model_dict(model_dict)

        return net
            

    def _init_output_scale(self, siren_cfg):

        scale_cfg = siren_cfg.get('output_scale', {})
        init = scale_cfg.get('init')
        
        # 1) set scale=1 (default)
        if init is None:
            output_scale = np.ones(self._n_outs)
            
        # 2) load from np file
        elif isinstance(init, str):
            output_scale = np.load(init)
        
        # 3) take from cfg as-it
        else:
            output_scale = np.asarray(init)
            
        assert len(output_scale)==self._n_outs, 'len(output_scale) != out_features'
        
        output_scale = torch.tensor(np.nan_to_num(output_scale), dtype=torch.float32)
        
        if scale_cfg.get('fix', True):
            self.register_buffer('output_scale', output_scale, persistent=True)
        else:
            self.register_parameter('output_scale', torch.nn.Parameter(output_scale))

