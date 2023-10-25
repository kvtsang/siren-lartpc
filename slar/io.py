import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from photonlib import PhotonLib
from slar.transform import partial_xform_vis

class PhotonLibDataset(Dataset):
    """
    PhotonLibrary in the form of torch Dataset for training Siren.
    Useful attributes:
      - plib ... PhotonLib instance
      - visibilities ... 1D array of visibility per voxel (in log scale if transform is enabled)
      - positions ... 1D array of a normalized (in the range -1 to 1 along each axis) position per voxel
    """
    
    def __init__(self, cfg):
        
        self.plib = PhotonLib.load(**cfg.get('photonlib',dict()))
        
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
        self.weights = 1
        weight_cfg = cfg['data']['dataset'].get('weight')
        if weight_cfg:
            print('[PhotonLibDataset] weighting the loss using',weight_cfg.get('method'))
            print('[PhotonLibDataset] params:', weight_cfg)
            if weight_cfg.get('method') == 'vis':
                self.weights = self.plib.vis * weight_cfg.get('factor', 1.)
                self.weights[self.weights<weight_cfg.get('threshold',1.e-8)] = 1.    
            else:
                raise NotImplementedError(f'The weight mode {weight_cfg.get("method")} is invalid.')
        
    def __len__(self):
        return len(self.plib.vis)
    
    def __getitem__(self, idx):
        output = dict(
            position=self.positions[idx],
            value=self.visibilities[idx],
            weight=self.weights[idx],
        )

        return output


