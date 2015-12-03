from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, tools
import os

from nbodykit import files
import xray

def load_data(root, box):
    
    toret = []
    dtype = [('r', float), ('corr', float), ('error', float)]
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
    def get_wg_poles(self):
        """
        Return a dictionary holding the configuration space galaxy spectrum 
        multipoles in redshift space, as used by the working group
        """
        try:
            return self._wg_poles
        except AttributeError:
            
            root = os.path.join(os.environ['RSD_DIR'], 'CorrelationFunction/Results')
            data = np.asarray([load_data(root, box) for box in self.boxes])
            
            ells = [0, 2]
            dims = ['box', 'rcen', 'ell']
            coords = {'box':self.boxes, 'rcen':data[0,:,0]['r'], 'ell':ells}
            d = {k:(dims, data[k]) for k in ['r', 'corr', 'error']}
            
            poles = xray.Dataset(d, coords=coords)
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

            basename = 'poles_challenge_box{box}_%s.dat' %tag
            path = os.path.join(root, 'corr_poles', basename)
            
            ells = [0, 2, 4]
            r, xi = [], []
            edges = None
            for box in boxes:

                d, meta = files.ReadPower1DPlainText(path.format(box=box))
                tostack = []
                for i, ell in enumerate(ells):
                    tostack.append(d[:,1+i])
                r.append([d[:,0]]*len(ells))
                xi.append(tostack)
                
                if edges is None: edges = meta['edges']

            r = np.asarray(r); xi = np.asarray(xi)
            rcen = 0.5*(edges[1:]+edges[:-1])
            dims = ['box', 'ell', 'rcen']
            coords = {'box':boxes, 'rcen':rcen, 'ell':ells}
            d = {'r':(dims, r), 'corr':(dims, xi)}
            
            poles = xray.Dataset(d, coords=coords)
            setattr(self, name, poles)
            return poles
            
        
    
