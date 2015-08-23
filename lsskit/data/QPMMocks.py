from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, io, tools
import os

class QPMMocks(PowerSpectraLoader):
    name = "QPMMocks"
    boxes = range(1, 991)
    
    def __init__(self, root, dk=None):
        self.root = root
        self.dk = dk
        
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
        
    #--------------------------------------------------------------------------
    # galaxy data
    #--------------------------------------------------------------------------
    def get_mean_Pgal(self, space='redshift', scaled=False):
        """
        Return the mean galaxy spectrum in redshift space
        """
        if space != 'redshift':
            raise NotImplementedError("only `redshift` space results exist for QPM mocks")
        tag = 'unscaled' if not scaled else 'scaled'
        name = '_mean_Pgal_%s_%s' %(tag, space)
        try:
            return getattr(self, name)
        except AttributeError:
            basename = 'pkmu_qpm_%s_990mean_0.6452_Nmu5.dat' %tag
            filename = os.path.join(self.root, space, basename)
            Pgal = io.load_data(filename)
            if self.dk is not None:
                Pgal = Pgal.reindex_k(self.dk, weights='modes', force=True)
            
            
            # add errors
            errs = (2./Pgal['modes'])**0.5 * Pgal['power']
            Pgal.add_column('error', errs)
            setattr(self, name, Pgal)
            return Pgal
            
    def get_Pgal(self, space='redshift', scaled=False):
        """
        Return the total galaxy spectrum in redshift space
        """
        if space != 'redshift':
            raise NotImplementedError("only `redshift` space results exist for QPM mocks")
        tag = 'unscaled' if not scaled else 'scaled'
        name = '_Pgal_%s_%s' %(tag, space)
        try:
            return getattr(self, name)
        except AttributeError:
            d = os.path.join(self.root, space)
            basename = 'pkmu_qpm_%s_{box:04d}_0.6452_Nmu5.dat' %tag
            coords = [self.boxes]
            Pgal = self.reindex(SpectraSet.from_files(d, basename, coords, ['box']), self.dk)
            
            # add the errors
            Pgal.add_errors()
            setattr(self, name, Pgal)
            return Pgal
            
    def get_covariance(self, space='redshift', scaled=False, **kwargs):
        """
        Return the covariance matrix from a set of QPM power spectra
        """
        Pgal = self.get_Pgal(space=space, scaled=scaled)
        _, _, _, C = tools.compute_pkmu_covariance(Pgal, **kwargs)
        return C
            