from . import io
from . import tools
from . import covariance
from .spectraset import SpectraSet, HaloSpectraSet

import numpy as np

class CombinedCorrFunction(dict):
    
    def __init__(self, low, high, r_switch=25.):
        
        super(CombinedCorrFunction, self).__init__()
        self.low      = low
        self.high     = high
        self.r_switch = r_switch
        
    @property
    def low_indices(self):
        return self.low['r'] <= self.r_switch
        
    @property
    def high_indices(self):
        return self.high['r'] > self.r_switch
        

    def __missing__(self, key):
        
        if key not in self.low or key not in self.high:
            raise ValueError("'%s' is not a valid column" %key)
            
        data = [self.low[key][self.low_indices], self.high[key][self.high_indices]]
        return np.concatenate(data)