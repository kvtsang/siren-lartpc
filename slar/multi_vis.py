from photonlib import AABox
from slar.nets import SirenVis
import torch
class MultiVis(torch.nn.Module):

    def __init__(self, cfg : dict=None):
        
        super().__init__()
        
        self._meta = AABox([[0.,0.],[0.,0.],[0.,0.]])
        self._model_v = []
        self._n_pmts = None
        
        if cfg is not None:
            self.configure(cfg)
            
            
    def configure(self, cfg : dict):

        self.config = dict(cfg)
        self.mvis_config   = cfg['multivis']
        weights     = self.mvis_config.get('weights',     [])
        weight_id   = self.mvis_config.get('weight_id',   [])
        input_scale = self.mvis_config.get('input_scale', [])
        aabox       = self.mvis_config.get('aabox',       [])
        
        if len(weights) and len(weight_id):
            # Validate config
            assert len(weight_id) == len(input_scale) == len(aabox), \
            f"weight_id ({len(weight_id)}), input_scale ({len(input_scale)}), aabox ({len(aabox)}) must be the same length."
            assert max(weight_id) < len(weights), f"weight_id max value ({max(weight_id)}) must be less than {len(weights)}"
            
            for w in weights:
                self.add_model(SirenVis.create_from_ckpt(w))
            
            aabox = torch.as_tensor(aabox)
            for i in range(len(weight_id)):
                m = AABox(aabox[i].T.reshape(3,2))
                self.add_meta(weight_id[i],m,input_scale[i])
                
            return
        
    @property
    def n_pmts(self):
        return self._n_pmts
        
    @property
    def meta(self):
        return self._meta

    @property
    def model_array(self):
        return self._model_v

    @property
    def device(self):
        '''
        Access the device this model is on. This function assumes all parameters are on the same device.

        Returns
        -------
        torch.device
            The device ID where this model instance resides on
        '''
        return next(self.model_array[0].parameters()).device

    def add_model(self, vis : SirenVis):
        if self._n_pmts is None:
            self._n_pmts = vis.n_pmts
        assert self.n_pmts == vis.n_pmts, f"the output shape mismatch: vis ({vis.n_pmts}) but expected {self.n_pmts}"
        model_name = 'model%d' % len(self._model_v)
        self.add_module(model_name,vis)
        self._model_v.append(getattr(self,model_name))

    def add_meta(self, model_idx:int, meta:AABox, input_scale:torch.Tensor=None):

        assert model_idx < len(self.model_array), f'model_index given ({model_idx}) is invalid (must be <{len(self.model_array)}'

        if input_scale:
            input_scale = torch.as_tensor(input_scale,dtype=torch.float32).clone().detach()
        else:
            input_scale = torch.ones(size=len(meta.ranges),dtype=torch.float32)
        assert len(input_scale) == len(meta.ranges), f'input_scale dimension must be {len(meta.ranges)}'

        # Append ranges
        if not hasattr(self,'children_meta'):
            self.register_buffer('children_meta',meta.ranges.clone().detach()[None,:],persistent=False)
        else:
        # make sure the given meta has no overlap with others
            for rs in self.children_meta:
                m=AABox(rs)
                if m.overlaps(meta):
                    raise ValueError(f'The provided meta \n{meta}overlaps with one of registered meta \n{m}')
            self.children_meta = torch.cat([self.children_meta,meta.ranges[None,:]])

        # Update own meta
        if self._meta is None:
            self._meta = meta
            self.register_buffer('meta_range',meta.ranges,persistent=False)
        else:
            self._meta.merge(meta)

        # Append the scaling factor
        if not hasattr(self,'input_scale'):
            self.register_buffer('input_scale',input_scale[None,:],persistent=False)
        else:
            self.input_scale = torch.cat([self.input_scale,input_scale[None,:]])

        # Append model id
        if not hasattr(self,'model_ids'):
            self.register_buffer('model_ids',torch.as_tensor([model_idx]),persistent=False)
        else:
            self.model_ids = torch.cat([self.model_ids,torch.as_tensor([model_idx])])
        
    def visibility(self, x):
        
        if len(x.shape) == 1:
            x = x[None,:]
        device = x.device

        vis = torch.zeros(size=(x.shape[0],self.n_pmts),dtype=torch.float32,device=self.device)
        
        for i,rs in enumerate(self.children_meta):
            m=AABox(rs)
            mask = m.contain(x)
            model = self.model_array[self.model_ids[i]]
            scale = self.input_scale[i]
            model.update_meta(ranges=m.ranges,input_scale=scale)
            vis[mask] = model.visibility(x[mask])
            
        return vis.to(device)

    
    def forward(self, x):
        
        if len(x.shape) == 1:
            x = x[None,:]
        device = x.device

        out = torch.zeros(x.shape[0],dtype=torch.float32,device=self.device)
        
        for i,rs in enumerate(self.children_meta):
            m = AABox(rs)
            mask = m.contain(x)
            model = self.model_array[self.model_ids[i]]
            scale = self.input_scale[i]
            model.update_meta(meta=m,input_scale=scale)
            out[mask] = model(m.norm_coord(x[mask]).to(self.device))
            
        return out.to(device)

    def model_dict(self, opt=None, epoch=-1):
        
        
        # It's possible to save all underlying models with self.state_dict()
        # But WE DO NOT DO THIS: because there are individual SirenVis attributes that cannot be retrieved.
        # Instead, we use each SirenVis function to store/restore, and store this model specific parameters by hand.        
        model_dict=dict(meta_range    = self.meta.ranges,
                        input_scale   = self.input_scale,
                        model_ids     = self.model_ids,
                        children_meta = self.children_meta,
                       )
        for i,model in enumerate(self._model_v):
            if model is None: continue
            print('[MultiVis] Saving model',i)
            model_dict['model%d'%i] = model.model_dict()
        if opt:
            model_dict['optimizer'] = opt.state_dict()
        if epoch>=0:
            model_dict['epoch'] = epoch
        return model_dict

    def save_state(self, filename, opt=None, epoch=-1):
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
        
        print('[MultiVis] saving the state ',filename)
        torch.save(self.model_dict(opt,epoch),filename)

    def load_model_dict(self, model_dict):
        '''
        Loads the network model and optimizer (and some hyper-) parameters from a dictionary
        Parameters
        ----------
        model_dict : dict
            Contains all model parameters necessary to re-instantiate the mode at the checkpoint.

        '''        
        from photonlib import AABox
        
        self._meta = AABox(model_dict['meta_range'])
        self.register_buffer('meta_range',    model_dict['meta_range'   ], persistent=False)
        self.register_buffer('input_scale',   model_dict['input_scale'  ], persistent=False)
        self.register_buffer('model_ids',     model_dict['model_ids'    ], persistent=False)
        self.register_buffer('children_meta', model_dict['children_meta'], persistent=False)

        for i in range(self.model_ids.max()+1):
            name='model%d' % i 
            if not i in self.model_ids:
                self._model_v.append(None)
                continue
            if not name in model_dict:
                raise KeyError(f'Model {name} not found !')
            print(f'[MultiVis] Creating SirenVis as {name}')
            model = SirenVis.create_from_model_dict(model_dict[name])
            self.add_module(name,model)
            self._model_v.append(getattr(self,name))

    def load_state(self, model_path):
        '''
        Loads the network model and optimizer (and some hyper-) parameters from a binary file.

        Parameters
        ----------
        model_path : str
            The checkpoint file name from which parameter values are loaded.

        '''

        print('[MultiVis] loading from checkpoint',model_path)
        with open(model_path, 'rb') as f:

            model_dict = torch.load(f, map_location='cpu')            

            self.load_model_dict(model_dict)
            
    @classmethod
    def create_from_ckpt(cls, model_path):
        print('[MultiVis] creating from checkpoint',model_path)
        with open(model_path, 'rb') as f:

            model_dict = torch.load(f, map_location='cpu')            

            self.load_model_dict(model_dict)