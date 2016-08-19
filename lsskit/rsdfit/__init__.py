class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
     
    def __getattribute__(self, key):
        try:
            return dict.__getattribute__(self, key)  
        except:
            return AttrDict()
       
# load the important modules`
import os

__config__ = ['RSDFIT_BIN', 'RSDFIT', 'RSDFIT_FITS', 'RSDFIT_DATA', 'RSDFIT_MODELS', 'RSDFIT_BATCH']

# default values are `None`
for c in __config__:
    globals()[c] = None

def _load_from_rc():
    """
    Load the configuration variables from a ``rsdfitrc`` file
    """
    filename = os.path.join(os.environ['HOME'], '.rsdfit', 'rsdfitrc')
    
    # if the config rc file exists, load values into globals()
    if os.path.isfile(filename):
        with open(filename) as f:
            code = compile(f.read(), filename, 'exec')
            exec(code, globals())
        
    
def _load_from_env():
    """
    Load the configuration variables from the environment
    """
    g = globals()
    for c in __config__:
        if g[c] is None and c in os.environ:
            g[c] = os.environ[c]
            
def _check_missing():
    """
    Crash if any of the configuration variables are `None`
    """
    g = globals()
    missing = []
    for c in __config__:
        if g[c] is None:
            missing.append(c)
            
    if len(missing):
        filename = os.path.join(os.environ['HOME'], '.rsdfit', 'rsdfitrc')
        msg = "please define the following variables via environment "
        msg += "variables or in the file '%s': %s" %(filename, str(missing))
        raise ValueError(msg)

# load the config variables
_load_from_rc()
_load_from_env()
_check_missing()

from .theory.base import BaseTheoryParams
from .data import PkmuDataParams, PoleDataParams
from .driver import DriverParams
from .runner import RSDFitRunner


            
    
        
        
