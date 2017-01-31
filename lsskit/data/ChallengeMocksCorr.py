from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, tools, io
import os
from nbodykit.dataset import DataSet

def make_edges(rcen):

    edges = np.empty(len(rcen)+1)
    spacing = 0.5*np.diff(rcen)
    edges[0], edges[-1] = rcen[0]-spacing[0], rcen[-1]+spacing[-1]
    edges[1:-1] = rcen[1:]-spacing
    return edges

def load_data(root, box):
    
    toret = []
    dtype = [('r', float), ('corr', float), ('error', float)]
    for kind in ['mono', 'quad']:
        data = np.loadtxt(os.path.join(root, '%s_Box%s_rescale.dat' %(kind, box)))
        C = np.loadtxt(os.path.join(root, 'covar_%s.dat' %kind)).reshape((data.shape[0], -1))
        
        errs = np.diag(C)**0.5
        data = np.concatenate([data, errs[:,None]], axis=1)
        edges = make_edges(data[:,0])
        data = np.squeeze(np.ascontiguousarray(data).view(dtype=numpy.dtype(dtype)))
        corr = DataSet(['r'], [edges], data)
        toret.append(corr)
        
    return toret
    
class ChallengeMocksCorr(PowerSpectraLoader):
    
    name = "ChallengeMocksCorr"
    boxes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    def __init__(self, root, dr=None):
        self.root = root
        self.dr = dr
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
    
    #--------------------------------------------------------------------------
    # multipoles data
    #--------------------------------------------------------------------------
    def get_wg_poles(self):
        """
        Return a dictionary holding the configuration space galaxy spectrum 
        multipoles in redshift space, as used by the working group
        """
        try:
            return self._wg_poles
        except AttributeError:
            import xarray
            
            root = os.path.join(os.environ['RSD_DIR'], 'CorrelationFunction/Results')
            data = np.asarray([load_data(root, box) for box in self.boxes])
            
            ells = [0, 2]
            dims = ['box', 'ell']
            coords = [self.boxes, ells]
            poles = xarray.DataArray(data, dims=dims, coords=coords)
            self._wg_poles = poles
            return poles
            
    def get_poles(self, scaled=True):
        """
        Return a dictionary holding the configuration space galaxy spectrum
        multipoles in redshift space, as measured by nbodykit
        """
        tag = '_scaled' if scaled else '_unscaled'
        name = '_poles' + tag
        try:
            return getattr(self, name)
        except AttributeError:

            basename = 'poles_challenge_box{box}%s.dat' %tag
            d = os.path.join(self.root, 'corr_poles')
                        
            loader = io.load_correlation
            kwargs = {'fields_to_sum':['RR', 'N']}
            poles = SpectraSet.from_files(loader, d, basename, [self.boxes], ['box'], args=('1d',), kwargs=kwargs)
            
            # reindex and add the errors
            poles = self.reindex(poles, 'r_cen', self.dr, weights='N')

            # now convert
            ells = [('corr_0',0), ('corr_2', 2), ('corr_4', 4)]
            poles = tools.unstack_multipoles(poles, ells, 'corr')
            setattr(self, name, poles)
            return poles
            
        
    
