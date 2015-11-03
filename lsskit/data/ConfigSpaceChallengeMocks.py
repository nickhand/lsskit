from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, tools
import os

def load_data(root, box):
    
    toret = []
    dtype = [('r', float), ('xi', float), ('err', float)]
    for kind in ['mono', 'quad']:
        data = np.loadtxt(os.path.join(root, '%s_Box%s_rescale.dat' %(kind, box)))
        C = np.loadtxt(os.path.join(root, 'covar_%s.dat' %kind)).reshape((data.shape[0], -1))
        
        errs = np.diag(C)**0.5
        data = np.concatenate([data, errs[:,None]], axis=1)
        data = np.squeeze(data.view(dtype=dtype))
        toret.append(data)
        
    return np.asarray(toret).T
    
class ConfigSpaceChallengeMocks(PowerSpectraLoader):
    name = "ConfigSpaceChallengeMocks"
    boxes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    def __init__(self, root):
        self.root = root
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
    
    #--------------------------------------------------------------------------
    # multipoles data
    #--------------------------------------------------------------------------
    def get_poles(self):
        """
        Return a dictionary holding the configuration space galaxy spectrum 
        multipoles in redshift space
        """
        name = '_poles'
        try:
            return getattr(self, name)
        except AttributeError:
            
            poles = {box:load_data(self.root, box) for box in self.boxes}            
            setattr(self, name, poles)
            return poles
    
