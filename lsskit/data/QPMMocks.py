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
    def get_mean_Pgal(self, spacing="", space='redshift', scaled=False, Nmu=5):
        """
        Return the mean galaxy spectrum in redshift space
        
        Parameters
        ----------
        spacing : str, optional
            the tag used to identify results with different k spacing, i.e. `dk005`
        space : {`real`, `redshift`}
            either return results in real or redshift space
        scaled : bool, optional
            return the results that have/haven't been scaled by AP factors
        Nmu : int, optional {`5`}
            the number of mu bins
        """
        if space != 'redshift':
            raise NotImplementedError("only `redshift` space results exist for QPM mocks")
        tag = 'unscaled' if not scaled else 'scaled'
        if spacing: spacing += "_"
        name = '_mean_Pgal_%s_%s%s_Nmu%d' %(tag, spacing, space, Nmu)
        try:
            return getattr(self, name)
        except AttributeError:
            
            # form the filename
            basename = 'pkmu_qpm_%s_990mean_0.6452_%sNmu%d.dat' %(tag, spacing, Nmu)
            filename = os.path.join(self.root, space, basename)
            
            # load the data and possibly re-index
            Pgal = io.load_data(filename)
            if self.dk is not None:
                Pgal = Pgal.reindex_k(self.dk, weights='modes', force=True)

            # add errors and return
            errs = (2./Pgal['modes'])**0.5 * Pgal['power']
            Pgal.add_column('error', errs)
            setattr(self, name, Pgal)
            return Pgal
            
    def get_Pgal(self, spacing="", space='redshift', scaled=False, Nmu=5):
        """
        Return the total galaxy spectrum in redshift space
        
        Parameters
        ----------
        spacing : str, optional
            the tag used to identify results with different k spacing, i.e. `dk005`
        space : {`real`, `redshift`}
            either return results in real or redshift space
        scaled : bool, optional
            return the results that have/haven't been scaled by AP factors
        Nmu : int, optional {`5`}
            the number of mu bins
        """
        if space != 'redshift':
            raise NotImplementedError("only `redshift` space results exist for QPM mocks")
        tag = 'unscaled' if not scaled else 'scaled'
        if spacing: spacing += "_"
        name = '_Pgal_%s_%s%s_Nmu%d' %(tag, spacing, space, Nmu)
        try:
            return getattr(self, name)
        except AttributeError:
            
            # form the filename and load the data
            d = os.path.join(self.root, space)
            basename = 'pkmu_qpm_%s_{box:04d}_0.6452_%sNmu%d.dat' %(tag, spacing, Nmu)
            coords = [self.boxes]
            Pgal = self.reindex(SpectraSet.from_files(d, basename, coords, ['box']), self.dk)
            
            # add the errors and return
            Pgal.add_errors()
            setattr(self, name, Pgal)
            return Pgal
            
    def get_covariance(self, spacing="", space='redshift', scaled=False, Nmu=5, **kwargs):
        """
        Return the covariance matrix from a set of QPM power spectra
        
        Parameters
        ----------
        spacing : str, optional
            the tag used to identify results with different k spacing, i.e. `dk005`
        space : {`real`, `redshift`}
            either return results in real or redshift space
        scaled : bool, optional
            return the results that have/haven't been scaled by AP factors
        """
        Pgal = self.get_Pgal(spacing=spacing, space=space, scaled=scaled, Nmu=Nmu)
        kwargs['return_extras'] = False
        return tools.compute_pkmu_covariance(Pgal, **kwargs)
            