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
    
    def __init__(self, cfg : dict, ckpt_file : str = None, meta = None):
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
            
            This mode is enabled if either ckpt_file (1st priority) or cfg['model']['ckpt_file']
            (2nd priority) is provided.
            
            Other than the model architecture, all attributes are loaded from the ckpt_file.
            This is to ensure the configuration used for training, and hence for the model stored
            in the ckpt_file, to remain the same after loading the state. 

        Parameters
        ----------
        cfg : dict
            Configuration parameters
        ckpt_file : str
            Checkpoint file. If provided, ignores some parameters in cfg. See the explanation above.
        '''

        siren_cfg = cfg['model']
        
        # initialize Siren class
        super().__init__(**siren_cfg['network'])

        self._n_outs = siren_cfg['network']['out_features']

        input_scale = torch.ones(size=(siren_cfg['network']['in_features'],),dtype=torch.float32)
        self.register_buffer('input_scale', input_scale)

        if siren_cfg.get('ckpt_file') or not ckpt_file is None:
            print('[SirenVis] loading the state from ckpt. Only "network" configuration used.)')
            if ckpt_file is None:
                ckpt_file = siren_cfg['ckpt_file']

            # register buffer/parameter for output_scale
            # actual values will be loaded from state_dict
            siren_init_cfg = siren_cfg.copy()
            siren_init_cfg.get('output_scale', {}).pop('init',None)
            self._init_output_scale(siren_init_cfg)

            self.load_state(ckpt_file)
            return

        # create meta
        if meta is not None:
            self._meta = meta
        else:
            self._meta = AABox.load(cfg['photonlib']['filepath'])

        # transform functions
        self._xform_cfg = cfg.get('transform_vis')
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self._xform_cfg)
        
        # extensions for visiblity model
        self._init_output_scale(siren_cfg)
        self._do_hardsigmoid = siren_cfg.get('hardsigmoid', False)


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
        '''
        Number of pmts, same interface as `PhontonLib.n_pmts`.

        Returns
        -------
        n_pmts: int
            Number of PMTs (i.e. number of output features)
        '''
        return self._n_outs

    def update_meta(self, meta:AABox, input_scale:torch.Tensor=None):
        self._meta = meta
        if input_scale:
            self.input_scale[:] = input_scale

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

        out = self(self.meta.norm_coord(x).to(self.device)).to(device)

        return self._inv_xform_vis(out)


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

        out = super().forward(x * self.input_scale)
        
        if self._do_hardsigmoid:
            out =  torch.nn.functional.hardsigmoid(out)
            
        out = out * self.output_scale
        
        return out


    def save_state(self, filename, opt=None, sch=None, epoch=0):
        '''
        Stores the network model and optimizer (and some hyper-) parameters to a binary file.


        Parameters
        ----------        
        filename : str
            The name of checkpoint file to be stored.
        opt : torch.optim.Optimizer
            The optimizer instance to store the optimizer state in the same file.
        sch : torch.optim.lr_scheduler, optional
            Learing rate scheduler instance to adjust lr (update every epoch)
        epoch : float
            The epoch count of training.
        '''
        
        state_dict=dict(epoch = epoch,
                        state_dict  = self.state_dict(),
                        xform_cfg   = self._xform_cfg,
                        aabox_ranges= self._meta.ranges,
                        do_hardsigmoid = self._do_hardsigmoid,
                       )

        if opt:
            state_dict['optimizer'] = opt.state_dict()

        if sch:
            state_dict['scheduler'] = sch.state_dict()

        print('[SirenVis] saving the state ',filename)
        torch.save(state_dict,filename)


    def load_state(self, model_path):
        '''
        Loads the network model and optimizer (and some hyper-) parameters from a binary file.

        Parameters
        ----------
        model_path : str
            The checkpoint file name from which parameter values are loaded.

        '''

        iteration = 0
        print('[SirenVis] loading the state',model_path)

        checkpoint = torch.load(model_path, map_location='cpu')            

        from photonlib import AABox
        self._meta = AABox(checkpoint['aabox_ranges'])
        self._xform_cfg = checkpoint['xform_cfg']
        self._xform_vis, self._inv_xform_vis = partial_xform_vis(self._xform_cfg)
        self._do_hardsigmoid = checkpoint['do_hardsigmoid']

        # 2024-03-10 kvt: for backward compatibility
        state_dict = checkpoint['state_dict']
        if 'input_scale' not in state_dict:
            state_dict['input_scale'] = self.input_scale
        if 'scale' in state_dict:
            state_dict['output_scale'] = state_dict.pop('scale')

        self.load_state_dict(state_dict)

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
            self.register_buffer('output_scale', output_scale)
        else:
            self.register_parameter('output_scale', torch.nn.Parameter(output_scale))

