import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from slar.transform import partial_xform_vis
from photonlib import PhotonLib

class PhotonLibDataset(Dataset):
    """
    PhotonLibrary in the form of torch Dataset for training Siren.

    """
    
    def __init__(self, cfg):
        '''
        Constructor

        Parameters
        ----------
        cfg : dict
            model configuration. Takes parameters for a function to transform visibilities to
            a log scale, and also the loss weighting scheme and parameters.

        Configuration
        -------------
        weight.method : str
            Currently only supported mode is "vis" (stands for visibility-based weighting).
            This means the high visibility voxels weights higher. The more the model makes
            mistakes at high visibility voxels, the more it gets penalized (i.e. it guides
            the model to learn high visibility voxels more). The logic behind is that the 
            higher visibility voxels are more rare compred to lower visibility voxels.
            Without weighting, the model can get high accuracy by simply predicting all 
            voxels are dark.

        weight.factor : float
            A scale-factor to be multiplied to the visibility value.

        weight.threshold : float
            The voxels with weights (=visibility * weight.factor) below this threshold value
            will have its weighting factor set to 1.0. 
        '''
        from photonlib import PhotonLib
        self.plib = PhotonLib.load(cfg)
        
        # tranform visiblity in pseudo-log scale (default: False)
        xform_params = cfg.get('transform_vis')
        if xform_params:
            print('[PhotonLibDataset] using log scale transformaion')
            print('[PhotonLibDataset] transformation params',xform_params)

        self.xform_vis, self.inv_xform_vis = partial_xform_vis(xform_params)
   
        self.visibilities = self.xform_vis(self.plib.vis)
        
        # transform the voxel ids to the normalized position (-1 to 1 scale along each axis)
        vox_ids = torch.arange(len(self.plib.vis))
        self.positions = self.plib.meta.norm_coord(self.plib.meta.voxel_to_coord(vox_ids))
        #self.positions = self.plib.meta.voxel_to_coord(vox_ids)
        
        # set the loss weighting factor matrix
        self.weights = torch.ones_like(self.visibilities)
        weight_cfg = cfg['data']['dataset'].get('weight')
        if weight_cfg:
            print('[PhotonLibDataset] weighting the loss using',weight_cfg.get('method'))
            print('[PhotonLibDataset] params:', weight_cfg)
            if weight_cfg.get('method') == 'vis':
                self.weights = self.plib.vis * weight_cfg.get('factor', 1.)
                self.weights[self.weights<weight_cfg.get('threshold',1.e-8)] = 1.    
            else:
                raise NotImplementedError(f'The weight mode {weight_cfg.get("method")} is invalid.')

        if 'device' in cfg['data']['dataset']:
            self.to(cfg['data']['dataset']['device'])

        
    def __len__(self):
        return len(self.plib.vis)

    def to(self,device):
        self.positions = self.positions.to(device)
        self.visibilities = self.visibilities.to(device)
        self.weights = self.weights.to(device)
        torch.cuda.synchronize()
        return self
    
    def __getitem__(self, idx):
        output = dict(
            position=self.positions[idx],
            value=self.visibilities[idx],
            weight=self.weights[idx],
        )

        return output

class PLibDataLoader:
    '''
    A fast implementation of PhotonLib dataloader.
    '''
    def __init__(self, cfg, device=None):
        '''
        Constructor.

        Arguments
        ---------
        cfg: dict
            Config dictionary. See "Examples" bewlow.

        device: torch.device (optional)
            Device for the returned data. Default: None.
        
        Examples
        --------
        This is an example configuration in yaml format.

        ```
		photonlib:
			filepath: plib_file.h5

		data:
			dataset:
				weight:
					method: vis
					factor: 1000000.0
					threshold: 1.0e-08
			loader:
				batch_size: 500
				shuffle: true

        transform_vis:
            eps: 1.0e-05
            sin_out: false
            vmax: 1.0
		```

        The `photonlib` section provide the input file of `PhotonLib`.

        [Optional] The `weight` subsection is the weighting scheme. Supported
        schemes are: 
        
        1. `vis`, where `weight ~ 1/vis * factor`.  Weights below `threshold`
        are set to one.  
        2. To-be-implemented.

        [Optional] The `loader` subsection mimics pytorch's `DataLoader` class,
        however, only `batch_size` and `shuffle` options are implemented.  If
        `loader` subsection is absent, the data loader returns the whole photon
        lib in a single entry.

        [Optional] The `transform_vis` subsection uses `log(vis+eps)` in the
        training. The final output is scaled to `[0,1]`.
        '''

        # load plib to device
        self._plib = PhotonLib.load(cfg).to(device)
        
        # get weighting scheme
        weight_cfg = cfg.get('data',{}).get('dataset',{}).get('weight')
        if weight_cfg:
            method = weight_cfg.get('method')
            if method == 'vis':
                self.get_weight = self.get_weight_by_vis
                print('[PLibDataLoader] weighting using', method)
                print('[PLibDataLoader] params:', weight_cfg)
            else:
                raise NotImplementedError(f'Weight method {method} is invalid')
            self._weight_cfg = weight_cfg
        else:
            self.get_weight = lambda vis : 1.

        # tranform visiblity in pseudo-log scale (default: False)
        xform_params = cfg.get('transform_vis')
        if xform_params:
            print('[PhotonLibDataset] using log scale transformaion')
            print('[PhotonLibDataset] transformation params',xform_params)

        self.xform_vis, self.inv_xform_vis = partial_xform_vis(xform_params)

        # prepare dataloader
        loader_cfg = cfg.get('data',{}).get('loader')
        self._batch_mode = loader_cfg is not None

        if self._batch_mode:
            # dataloader in batches
            self._batch_size = loader_cfg.get('batch_size', 1)
            self._shuffle = loader_cfg.get('shuffle', False)
        else:
            # returns the whole plib in a single batch
            n_voxels = len(self._plib)
            vox_ids = torch.arange(n_voxels, device=device)

            meta = self._plib.meta
            pos = meta.norm_coord(meta.voxel_to_coord(vox_ids))

            vis = self._plib.vis
            w = self.get_weight(vis)

            self._cache = dict(position=pos, value=vis, weight=w)

    @property
    def device(self):
        return self._plib.device
    
    def get_weight_by_vis(self, vis):
        '''
        Weight by inverse visibility, `weight  = 1/vis * factor`.
        Weights below `threshold` are set to 1.

        Arguments
        ---------
        vis: torch.Tensor
            Visibility values.

        Returns
        -------
        w: trorch.Tensor
            Weight values with `w.shape == vis.shape`.
        '''
        factor = self._weight_cfg.get('factor', 1.)
        threshold = self._weight_cfg.get('threshold', 1e-8)
        w = vis * factor
        w[w<threshold] = 1.
        return w
        
    def __len__(self):
        '''
        Number of batches.
        '''
        from math import ceil
        if self._batch_mode:
            return ceil(len(self._plib) / self._batch_size)

        return 1
        
    def __iter__(self):
        '''
        Generator of batch data.

        For non-batch mode, the whole photon lib is returned in a single entry
        from the cache.
        '''
        if self._batch_mode:
            meta = self._plib.meta
            n_voxels = len(self._plib)
            if self._shuffle:
                vox_list = torch.randperm(n_voxels, device=self.device)
            else:
                vox_list = torch.arange(n_voxels, device=self.device)

            for b in range(len(self)):
                sel = slice(b*self._batch_size, (b+1)*self._batch_size)

                vox_ids = vox_list[sel]
                pos = meta.norm_coord(meta.voxel_to_coord(vox_ids))
                vis = self._plib[vox_ids]
                w = self.get_weight(vis)
                
                output = dict(position=pos, value=vis, weight=w)
                yield output
        else:
            yield self._cache
