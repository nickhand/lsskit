import os
import numpy as np

# load the config file
config_filename = os.path.join(os.path.curdir, 'fits.rc')
if not os.path.isfile(config_filename):
    raise ValueError("please specify configuration file: '%s'" %config_filename)

# load the configuration
model_kmax = None
execfile(config_filename, globals())


def rsd_model():
    """
    Load and return the RSD model
    """
    if '_rsd_model' in globals():
        return globals()['_rsd_model']
            
    model_path = os.path.join(os.environ['RSDFIT_MODELS'], model_name)
    if not os.path.exists(model_path):
        raise ValueError("no such model file: '%s'" %model_path)
                
    global _rsd_model
    _rsd_model = np.load(model_path).tolist()
    return _rsd_model

from .core import FittingSet
from .plot import *
from . import table