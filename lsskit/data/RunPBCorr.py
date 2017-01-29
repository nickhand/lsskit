from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, tools, io
import os
    
class RunPBCorr(PowerSpectraLoader):
    
    name = "RunPBCorr"
    a = ['0.5000', '0.5714', '0.6061', '0.6452', '0.6667', '0.6897', '0.7143', '0.8000', '0.9091', '1.0000']
    box = ['%02d' %i for i in range(10)]
    mass = range(8)
    
    def __init__(self, root, dr=None):
        self.root = root
        self.dr = dr
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
    
    #--------------------------------------------------------------------------
    # 1D data
    #--------------------------------------------------------------------------            
    def get_1d_matter(self, kind='small_r', average='box'):
        """
        Return the 1D real-space matter correlation functions, with errors
        estimated from the scatter in the 10 realization
        """
        valid_kinds = ['small_r', 'fft']
        if kind not in valid_kinds:
            raise ValueError("valid choices for `kind` are: %s" %str(valid_kinds))
        
        name = '_1d_matter_%s' %kind
        if average is not None:
            name += '_'+average

        try:
            return getattr(self, name)
        except AttributeError:

            basename = 'corr_1d_runPB_PB{box}_{a}_%s.dat' %kind
            d = os.path.join(self.root, 'matter/config/real')

            loader = io.load_correlation
            coords = [self.a, self.box]; dims = ['a', 'box']
            corr = SpectraSet.from_files(loader, d, basename, coords, dims, args=('1d',))
            
            # reindex and add the errors
            corr = self.reindex(corr, 'r_cen', self.dr, weights='N')
            
            # compute errors
            errs = {}
            for a in corr['a'].values:
                data = []
                for box in corr['box'].values:
                    xi = corr.sel(box=box, a=a).get()
                    data.append(xi['corr'])
                    
                errs[a] = np.diag(np.cov(np.asarray(data).T))**0.5
                if average is not None:
                    errs[a] /= (len(self.box))**0.5
                
            # average?
            if average is not None:
                corr = corr.average(axis=average)

            
            # add the errors
            for key in corr.ndindex():
                xi = corr.loc[key].get()
                xi['error'] = errs[key['a']]
    
            setattr(self, name, corr)
            return corr
        
    def get_1d_halo_matter(self, kind='small_r', average='box'):
        """
        Return the 1D real-space halo-matter correlation functions, 
        with errors estimated from the scatter in the 10 realizations
        """
        valid_kinds = ['small_r', 'fft']
        if kind not in valid_kinds:
            raise ValueError("valid choices for `kind` are: %s" %str(valid_kinds))
            
        name = '_1d_halo_matter_%s' %kind
        if average is not None:
            name += '_'+average

        try:
            return getattr(self, name)
        except AttributeError:

            basename = 'corr_1d_hm{mass}_runPB_PB{box}_{a}_%s.dat' %kind
            d = os.path.join(self.root, 'halo-matter/config/real')

            loader = io.load_correlation
            coords = [self.a, self.mass, self.box]; dims = ['a', 'mass', 'box']
            corr = SpectraSet.from_files(loader, d, basename, coords, dims, args=('1d',))
        
            # reindex and add the errors
            corr = self.reindex(corr, 'r_cen', self.dr, weights='N')
        
            # compute errors
            errs = {}
            for key in corr.ndindex(dims=['a', 'mass']):
                xi = corr.sel(**key)
                
                data = []
                for x in xi:
                    x = x.get()
                    data.append(x['corr'])
                
                subkey = (key['a'], key['mass'])
                errs[subkey] = np.diag(np.cov(np.asarray(data).T))**0.5
                if average is not None:
                    errs[subkey] /= (len(self.box))**0.5
            
            # average?
            if average is not None:
                corr = corr.average(axis=average)
        
            # add the errors
            for key in corr.ndindex():
                xi = corr.loc[key].get()
                xi['error'] = errs[(key['a'], key['mass'])]

            setattr(self, name, corr)
            return corr
            
        
    
