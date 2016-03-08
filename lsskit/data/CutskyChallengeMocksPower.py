import os

from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, tools, io
from nbodykit.dataset import Power1dDataSet


class CutskyChallengeMocksPower(PowerSpectraLoader):
    name = "CutskyChallengeMocksPower"
    boxes = range(1, 85)
    
    def __init__(self, root, dk=None):
        self.root = root
        self.dk = dk
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
        
        
    def get_poles(self, tag="", scaled=False, average=False):
        """
        Return the cutsky galaxy multipoles in redshift space, as
        measured by `nbodykit`
        """
        scaled_tag = 'scaled' if scaled else 'unscaled'
        name = '_poles_' + scaled_tag
        if average: name += '_mean'
        if tag: 
            tag = '_'+tag
            name += tag
        
        try:
            return getattr(self, name)         
        except AttributeError:
        
            # form the filename and load the data
            d = os.path.join(self.root, 'nbodykit/poles')
            basename = 'poles_cutskyN{box:d}_%s_no_fkp_dk005%s.dat' %(scaled_tag, tag)

            # read in the data
            loader = io.load_power
            mapcols = {'power_0.real':'mono', 'power_2.real':'quad', 'power_4.real':'hexadec'}
            usecols = ['k', 'mono', 'quad', 'hexadec', 'modes']
            kwargs = {'usecols':usecols, 'mapcols':mapcols}
            poles = SpectraSet.from_files(loader, d, basename, [self.boxes], ['box'], args=('1d',), kwargs=kwargs)
        
            # reindex
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')
            
            # unstack the poles
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            poles = tools.unstack_multipoles(poles, ells, 'power')
            
            # compute errors
            errs = {}
            for ell in poles['ell'].values:
                data = []
                for box in poles['box'].values:
                    p = poles.sel(box=box, ell=ell).values
                    data.append(p['power'])

                errs[ell] = np.diag(np.cov(np.asarray(data).T))**0.5
                if average:
                    errs[ell] /= (len(self.boxes))**0.5
            
            # average?
            if average:
                poles = poles.average(axis='box')
                
            # add the errors
            for key in poles.ndindex():
                p = poles.loc[key].values
                p['error'] = errs[key['ell']]
        
            setattr(self, name, poles)
            return poles
            

    def get_florian_poles(self, scaled=True, average=False):
        """
        Return the cutsky galaxy multipoles in redshift space, as
        measured by Florian's code
        """
        scaled_tag = 'scaled' if scaled else 'unscaled'
        name = '_poles_florian_' + scaled_tag
        if average: name += '_mean'
        
        try:
            return getattr(self, name)         
        except AttributeError:
        
            # form the filename and load the data
            d = os.path.join(self.root, 'data', scaled_tag)
            if scaled:
                basename = 'bianchips_cutsky_TSC_0.7_{box:d}.dat'
            else:
                basename = 'bianchips_Ncutsky_{box:02d}.dat'

            # read in the data
            loader = io.read_cutsky_power_poles
            kwargs = {'skiprows' : 31, 'sum_only':['modes'], 'force_index_match':True}
            poles = SpectraSet.from_files(loader, d, basename, [self.boxes], ['box'], kwargs=kwargs)
        
            # reindex
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')
            
            # unstack the poles
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            poles = tools.unstack_multipoles(poles, ells, 'power')
            
            if average:
                poles = poles.average(axis='box', weights='modes')
        
            setattr(self, name, poles)
            return poles
            
    def get_window(self):
        """
        Return the formatted window function for the cutsky challenge mocks
        """
        
        filename = os.path.join(self.root, 'extra', 'wilson_random_win_Ncutsky_0.4_strim_7.00e+02_smooth_201x2.dat')
        return np.loadtxt(filename)