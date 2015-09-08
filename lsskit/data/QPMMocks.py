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
    # galaxy pkmu data
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
            filename = os.path.join(self.root, space, 'pkmu', basename)
            
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
            d = os.path.join(self.root, space, 'pkmu')
            basename = 'pkmu_qpm_%s_{box:04d}_0.6452_%sNmu%d.dat' %(tag, spacing, Nmu)
            coords = [self.boxes]
            Pgal = self.reindex(SpectraSet.from_files(d, basename, coords, ['box']), self.dk)
            
            # add the errors and return
            Pgal.add_errors()
            setattr(self, name, Pgal)
            return Pgal
            
    #--------------------------------------------------------------------------
    # galaxy multipoles data
    #--------------------------------------------------------------------------
    def get_mean_poles(self, spacing="dk005", space='redshift', scaled=False,):
        """
        Return the mean galaxy spectrum multipoles (mono, quad, hexadec) in 
        redshift space
        """
        if space != 'redshift':
            raise NotImplementedError("only `redshift` space results exist for QPM mocks")
        tag = 'unscaled' if not scaled else 'scaled'
        if spacing: spacing = '_'+spacing
        name = '_mean_poles_%s_%s%s' %(tag, space, spacing)
        try:
            return getattr(self, name)
        except AttributeError:
            data = []
            poles = ['mono', 'quad', 'hexadec']
            
            for pole in poles:
                basename = '%s_qpm_%s_990mean_0.6452%s.dat' %(pole, tag, spacing)
                filename = os.path.join(self.root, space, 'poles', basename)
            
                P = io.load_data(filename)
                if self.dk is not None:
                    P = P.reindex_k(self.dk, weights='modes', force=True)
                data.append(P)
                
            toret = SpectraSet(data, coords=[[0, 2, 4]], dims=['ell'])
            setattr(self, name, toret)
            return toret
    
    def get_poles(self, spacing="dk005", space='redshift', scaled=False):
        """
        Return the galaxy multipoles in redshift space
        """
        if space != 'redshift':
            raise NotImplementedError("only `redshift` space results exist for QPM mocks")
        tag = 'unscaled' if not scaled else 'scaled'
        if spacing: spacing = "_" +spacing
        name = '_poles_%s_%s%s' %(tag, space, spacing)
        try:
            return getattr(self, name)
        except AttributeError:
            
            # form the filename and load the data
            d = os.path.join(self.root, space, 'poles')
            basename = '{ell}_qpm_%s_{box:04d}_0.6452%s.dat' %(tag, spacing)
            coords = [self.boxes, ['mono', 'quad', 'hexadec']]
            P = self.reindex(SpectraSet.from_files(d, basename, coords, ['box', 'ell']), self.dk)
            
            P.coords['ell'] = [0, 2, 4]
            setattr(self, name, P)
            return P
    
    #--------------------------------------------------------------------------
    # covariances
    #--------------------------------------------------------------------------
    def get_pkmu_covariance(self, spacing="dk005", space='redshift', scaled=False, Nmu=5, **kwargs):
        """
        Return the P(k,mu) covariance matrix from a set of QPM power spectra
        
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
        
        
    def get_pole_covariance(self, spacing="dk005", space='redshift', scaled=False, **kwargs):
        """
        Return the multipoles covariance matrix from a set of QPM power spectra
        
        Parameters
        ----------
        spacing : str, optional
            the tag used to identify results with different k spacing, i.e. `dk005`
        space : {`real`, `redshift`}
            either return results in real or redshift space
        scaled : bool, optional
            return the results that have/haven't been scaled by AP factors
        """
        poles = self.get_poles(spacing=spacing, space=space, scaled=scaled)
        kwargs['return_extras'] = False
        return tools.compute_pole_covariance(poles, [0, 2, 4], **kwargs)
            