import sys, os, importlib, glob
import torch
from functools import partial


def list_available_devices():

    devs=dict(cpu=torch.device('cpu'))

    if torch.cuda.is_available():
        devs['cuda'] = torch.device('cuda:0')
        
    if torch.backends.mps.is_available():
        devs['mps'] = torch.device('mps')

    return devs


def get_device(request):

    devs = list_available_devices()

    if not request in devs:
        print(request,'not supported')
        return None
    else:
        return devs[request]


def import_from(src):
    if src.count('.') == 0:
        module = sys.modules['__main__']
        obj_name = src
    else:
        module_name, obj_name = os.path.splitext(src)
        module = importlib.import_module(module_name)

    return getattr(module, obj_name.lstrip('.'))



def get_config_dir():

    return os.path.join(os.path.dirname(__file__),'config')


def list_config(full_path=False):

    fs = glob.glob(os.path.join(get_config_dir(), '*.yaml'))

    if full_path:
        return fs

    return [os.path.basename(f)[:-5] for f in fs]


def get_config(name):

    options = list_config()
    results = list_config(True)

    if name in options:
        return results[options.index(name)]

    alt_name = name + '.yaml'
    if alt_name in options:
        return results[options.index(alt_name)]

    print('No data found for config name:',name)
    raise NotImplementedError



class CSVLogger:

    def __init__(self,cfg):
        
        log_cfg = cfg.get('logger',dict())
        self._logdir  = self.make_logdir(log_cfg.get('dir_name','logs'))
        self._logfile = os.path.join(self._logdir, cfg.get('file_name','log.csv'))
        self._log_every_nsteps = log_cfg.get('log_every_nsteps',1)

        print('[CSVLogger] output log directory:',self._logdir)
        print(f'[CSVLogger] recording a log every {self._log_every_nsteps} steps')
        self._fout = None
        self._str  = None
        self._dict = {}
        
        self._analysis_dict={}
        
        for key, kwargs in log_cfg.get('analysis',dict()).items():
            print('[CSVLogger] adding analysis function:',key)
            self._analysis_dict[key] = partial(getattr(importlib.import_module('slar.analysis'),key), **kwargs)
        
    @property
    def logfile(self):
        return self._logfile
    
    @property
    def logdir(self):
        return self._logdir
        
    def make_logdir(self, dir_name):
        versions = [int(d.split('-')[-1]) for d in glob.glob(os.path.join(dir_name,'version-[0-9][0-9]'))]
        ver = 0
        if len(versions):
            ver = max(versions)+1
        logdir = os.path.join(dir_name,'version-%02d' % ver)
        os.makedirs(logdir)

        return logdir

    def record(self, keys, vals):
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]
            
    def step(self, iteration, label=None, pred=None):
        if not iteration % self._log_every_nsteps == 0:
            return
        
        if not None in (label,pred):
            for key, f in self._analysis_dict.items():
                self.record([key],[f(label,pred)])
        self.write()

    def write(self):
        if self._str is None:
            self._fout=open(self._logfile,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'
        self._fout.write(self._str.format(*(self._dict.values())))
        self.flush()
        
    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()
