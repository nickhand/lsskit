import os

from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, utils

def to_pkresult(filename, skiprows=31):
    """
    Return a list of `PkResult` objects for each multipole in
    the input data file
    """
    from nbodykit import pkresult
    data = np.loadtxt(filename, skiprows=skiprows, comments=None)
    
    # make the edges
    dk = 0.005
    lower = data[:,0]-dk/2.
    upper = data[:,0]+dk/2.
    edges = np.array(zip(lower, upper))
    edges = np.concatenate([edges.ravel()[::2], [edges[-1,-1]]])

    toret = []
    for i, ell in enumerate([0, 2, 4]):
    
        d = data[:,[1, 2+i, -1]]
        pk = pkresult.PkResult.from_dict(d, ['k', 'power', 'modes'], sum_only=['modes'], edges=edges)
        toret.append(pk)
    return toret

class CutskyChallengeMocks(PowerSpectraLoader):
    name = "CutskyChallengeMocks"
    boxes = range(1, 84)
    
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
            coords = [self.boxes]
            dims = ['box']
            ells = [0, 2, 4]

            # read in the data
            data = []
            for i, f in utils.enum_files(d, basename, dims, coords, ignore_missing=False):
                if f is not None:
                    data.append(to_pkresult(f))
                else:
                    data.append([np.nan]*len(ells))
            
            coords += [ells]
            dims += ['ell']
            poles = self.reindex(SpectraSet(data, coords=coords, dims=dims), self.dk)
            
            setattr(self, name, poles)
            return poles