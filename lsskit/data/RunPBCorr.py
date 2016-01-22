from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, tools, io
import os
from nbodykit.dataset import Corr1dDataSet
    
class RunPBCorr(PowerSpectraLoader):
    
    name = "RunPBCorr"
    a = ['0.5000', '0.5714', '0.6061', '0.6452', '0.6667', '0.6897', '0.7143', '0.8000', '0.9091', '1.0000']
    mass = range(8)
    
    def __init__(self, root, realization='10mean', dr=None):
        self.root = root
        self.dr = dr
        self.tag = realization
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
    
    #--------------------------------------------------------------------------
    # multipoles data
    #--------------------------------------------------------------------------            
    def get_1d_halo_matter(self, space='real'):
        """
        Return a dictionary holding the configuration space halo - matter
        cross correlation
        """
        if space != 'real':
            raise ValueError("only `real` space results available")
            
        name = '_hm_%s_poles' %space
        try:
            return getattr(self, name)
        except AttributeError:

            basename = 'poles_hm{mass}_runPB_%s_{a}.dat' %self.tag
            d = os.path.join(self.root, 'halo-matter', space, 'corr_poles')
                        
            loader = io.load_correlation
            kwargs = {'sum_only':['N'], 'force_index_match':True}
            coords = [self.a, self.mass]; dims = ['a', 'mass']
            corr = SpectraSet.from_files(loader, d, basename, coords, dims, args=('1d',), kwargs=kwargs)
            
            # reindex and add the errors
            corr = self.reindex(corr, 'r_cen', self.dr, weights='N')

            corr.add_corr_errors()
            setattr(self, name, corr)
            return corr
            
        
    
