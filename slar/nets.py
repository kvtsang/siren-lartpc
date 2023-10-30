import torch
import numpy as np

from photonlib.meta import AABox
from slar.base import Siren
from slar.transform import partial_xform_vis

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
    
    def __init__(self, cfg : dict, ckpt_file : str = None):
        '''
        Constructor takes two different modes depending on the arguments.

        Mode 1 : built entirely from the configuration file
            
            This mode is taken if both ckpt_file and cfg['model']['ckpt_file'] are None
            
            The model constructs all attributes based on the configuration file. In particular,
            this includes the function input coordinate information (self._meta), visibility
            forward and inverse transformation functions (self._xform_vis, self._inv_xform_vis),
            a flag to apply sigmoid on the output of Siren (self._do_hardsigmoid), and the scaling
            factor applied to each visibility (self.scale).

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

        if siren_cfg.get('ckpt_file') or not ckpt_file is None:
            print('[SirenVis] loading the state from ckpt. Only "network" configuration used.)')
            if ckpt_file is None:
                ckpt_file = siren_cfg['ckpt_file']
            self.load_state(ckpt_file)
            return

        # create meta
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

        out = self(self.meta.norm_coord(x))

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

        out = super().forward(x)
        
        if self._do_hardsigmoid:
            out =  torch.nn.functional.hardsigmoid(out)
            
        out = out * self.scale
        
        return out


    def save_state(self, filename, opt=None, epoch=0):
        '''
        Stores the network model and optimizer (and some hyper-) parameters to a binary file.


        Parameters
        ----------        
        filename : str
            The name of checkpoint file to be stored.
        opt : torch.optim.Optimizer
            The optimizer instance to store the optimizer state in the same file.
        epoch : float
            The epoch count of training.
        '''
        
        state_dict=dict(epoch = epoch,
                        state_dict  = self.state_dict(),
                        xform_cfg   = self._xform_cfg,
                        aabox_ranges= self._meta.ranges,
                        do_hardsigmoid = self._do_hardsigmoid
                       )
        # check if scale should be saved
        pnames = [ name for name, p in self.named_parameters()]
        if not 'scale' in pnames:
            state_dict['scale'] = self.scale 

        if opt:
            state_dict['optimizer'] = opt.state_dict()

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
        with open(model_path, 'rb') as f:

            checkpoint = torch.load(f, map_location='cpu')            

            self._meta = AABox(checkpoint['aabox_ranges'])

            self._xform_cfg = checkpoint['xform_cfg']
            self._xform_vis, self._inv_xform_vis = partial_xform_vis(self._xform_cfg)

            self._do_hardsigmoid = checkpoint['do_hardsigmoid']

            scale = torch.ones(self._n_outs,dtype=torch.float32)

            if 'scale' in checkpoint.keys():
                self.register_buffer('scale', scale)
                self.scale = checkpoint['scale']
            else:
                self.register_parameter('scale', torch.nn.Parameter(scale))

            self.load_state_dict(checkpoint['state_dict'])
            

    def _init_output_scale(self, siren_cfg):

        scale_cfg = siren_cfg.get('output_scale', {})
        init = scale_cfg.get('init')
        
        # 1) set scale=1 (default)
        if init is None:
            scale = np.ones(self._n_outs)
            
        # 2) load from np file
        elif isinstance(init, str):
            scale = np.load(init)
        
        # 3) take from cfg as-it
        else:
            scale = np.asarray(init)
            
        assert len(scale)==self._n_outs, 'len(output_scale) != out_features'
        
        scale = torch.tensor(np.nan_to_num(scale), dtype=torch.float32)
        
        if scale_cfg.get('fix', True):
            self.register_buffer('scale', scale)
        else:
            self.register_parameter('scale', torch.nn.Parameter(scale))





