import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from slar.transform import partial_xform_vis

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


