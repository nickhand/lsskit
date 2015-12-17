import os

from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, tools
from nbodykit.dataset import Power1dDataSet

def to_dataset(filename, skiprows=31):
    """
    Return a list of `Power1dDataSet` objects for each multipole in
    the input data file
    """
    data = np.loadtxt(filename, skiprows=skiprows, comments=None)
    
    # make the edges
    dk = 0.005
    lower = data[:,0]-dk/2.
    upper = data[:,0]+dk/2.
    edges = np.array(zip(lower, upper))
    edges = np.concatenate([edges.ravel()[::2], [edges[-1,-1]]])

    toret = []
    columns = ['k', 'mono', 'quad', 'hexadec', 'modes']
    meta = {'edges':edges, 'sum_only':['modes'], 'force_index_match':True}
    d = data[:,[1, 2, 3, 4, -1]]
    
    return Power1dDataSet.from_nbkit(d, meta, columns=columns)


class CutskyChallengeMocksPower(PowerSpectraLoader):
    name = "CutskyChallengeMocksPower"
    boxes = range(1, 85)
    
    def __init__(self, root, dk=None):
        self.root = root
        self.dk = dk
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
        
    #--------------------------------------------------------------------------
    # multipoles data
    #--------------------------------------------------------------------------
    def get_mean_poles(self, spacing="dk005", space='redshift', scaled=False, Nmu=100):
        """
        Return the mean of the cutsky galaxy multipoles in redshift space
        """
        name = '_mean_poles'
        try:
            return getattr(self, name)
        except AttributeError:
        
            # read in the data
            d = os.path.join(self.root, 'data')
            basename = 'bianchips_cutsky_TSC_0.7_84mean.dat'
            f = os.path.join(d, basename)
            data = to_pkresult(f, skiprows=0)
            
            coords = [[0, 2, 4]]
            dims = ['ell']
            poles = self.reindex(SpectraSet(data, coords=coords, dims=dims), self.dk)

            setattr(self, name, poles)
            return poles

    def get_poles(self):
        """
        Return the cutsky galaxy multipoles in redshift space
        """
        name = '_poles'
        try:
            return getattr(self, name)            
        except AttributeError:
        
            # form the filename and load the data
            d = os.path.join(self.root, 'data')
            basename = 'bianchips_cutsky_TSC_0.7_{box:d}.dat'

            # read in the data
            kwargs = {'skiprows' : 31}
            poles = SpectraSet.from_files(to_dataset, d, basename, [self.boxes], ['box'], kwargs=kwargs)
        
            # reindex
            poles = self.reindex(poles, 'k_cen', self.dk)
            
            # unstack the poles
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            poles = tools.unstack_multipoles(poles, ells, 'power')
        
            setattr(self, name, poles)
            return poles