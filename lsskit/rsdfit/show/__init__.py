import os
import numpy as np

# load the config file
config_filename = os.path.join(os.path.curdir, 'fits.rc')
if not os.path.isfile(config_filename):
    raise ValueError("please specify configuration file: '%s'" %config_filename)

# load the files
execfile(config_filename, globals())


def get_model():
    """
    Load and return the RSD model
    """
    model_path = os.path.join(os.environ['RSDFIT_MODELS'], model_name)
    if not os.path.exists(model_path):
        raise ValueError("no such model file: '%s'" %model_path)
                
    return np.load(model_path).tolist()
            
rsd_model = get_model()
model_kmax = None
if 'model_kmax' in globals():
    rsd_model.kmax = model_kmax

from .core import FittingSet
from .plot import *
from . import table