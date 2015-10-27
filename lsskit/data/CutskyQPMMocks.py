import os

from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, covariance, utils

def to_pkresult(filename):
    """
    Return a list of `PkResult` objects for each multipole in
    the input data file
    """
    from nbodykit import pkresult
    data = np.loadtxt(filename, skiprows=31, comments=None)
    
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
    
    
class CutskyQPMMocks(PowerSpectraLoader):
    name = "CutskyQPMMocks"
    boxes = range(1, 1001)
    
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
            basename = 'bianchips_qpmdr12_TSC_964mean.dat'
            f = os.path.join(self.root, basename)
            data = to_pkresult(f)
            
            coords = [[0, 2, 4]]
            dims = ['ell']
            poles = self.reindex(SpectraSet(data, coords=coords, dims=dims), self.dk)

            setattr(self, name, poles)
            return poles

    def get_poles(self, remove_null=False):
        """
        Return the cutsky galaxy multipoles in redshift space
        
        Parameters
        ----------
        remove_null : bool, optional (`False`)
            If `True`, remove any null entries before returning
        """
        name = '_poles'
        try:
            poles = getattr(self, name)
            if remove_null:
                idx = np.all(poles.notnull(), axis=-1)
                poles = poles[idx,:]
            return poles
            
        except AttributeError:
        
            # form the filename and load the data
            basename = 'bianchips_qpmdr12_TSC_{box:04d}.dat'
            coords = [self.boxes]
            dims = ['box']
            ells = [0, 2, 4]

            # read in the data
            data = []
            for i, f in utils.enum_files(self.root, basename, dims, coords, ignore_missing=True):
                if f is not None:
                    data.append(to_pkresult(f))
                else:
                    data.append([np.nan]*len(ells))
            
            coords += [ells]
            dims += ['ell']
            poles = self.reindex(SpectraSet(data, coords=coords, dims=dims), self.dk)
            setattr(self, name, poles)
            
            if remove_null:
                idx = np.all(poles.notnull(), axis=-1)
                poles = poles[idx,:]
            return poles
            
    #--------------------------------------------------------------------------
    # covariance
    #--------------------------------------------------------------------------        
    def get_pole_covariance(self, **kwargs):
        """
        Return the multipoles covariance matrix from a set of cutsky 
        QPM power spectra
        """
        # get the non-null poles
        poles = self.get_poles(remove_null=True)

        kwargs['extras'] = False
        return covariance.compute_pole_covariance(poles, [0, 2, 4], **kwargs)
            