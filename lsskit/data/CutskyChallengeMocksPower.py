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
        

    #--------------------------------------------------------------------------
    # multipoles
    #--------------------------------------------------------------------------
    def get_poles(self, tag="", scaled=False, average=False, subtract_shot_noise=True):
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
            
            
            if subtract_shot_noise:
                for key in poles.ndindex():
                    if key['ell'] == 0:
                        p = poles.loc[key].values
                        p['power'] = p['power'] - p.attrs['shot_noise']
            
            # average?
            if average:
                poles = poles.average(axis='box')
                
            # add the errors
            for key in poles.ndindex():
                p = poles.loc[key].values
                p['error'] = errs[key['ell']]
        
            setattr(self, name, poles)
            return poles
            
    def get_my_poles(self, scaled=False, average=False, tag="", subtract_shot_noise=True):
        """
        Return the cutsky galaxy multipoles in redshift space, measured
        for my 84 box realizations
        """ 
        scaled_tag = 'scaled' if scaled else 'unscaled'
        name = '_my_poles_' + scaled_tag
        if average: name += '_mean'
        
        if tag: 
            tag = '_'+tag
            name += tag
        
        try:
            return getattr(self, name)         
        except AttributeError:
        
            # form the filename and load the data
            d = os.path.join(self.root, 'nbodykit/poles')
            basename = 'poles_my_cutskyN{box:d}_%s_no_fkp_dk005%s.dat' %(scaled_tag, tag)

            # read in the data
            loader = io.load_power
            mapcols = {'power_0.real':'mono', 'power_2.real':'quad', 'power_4.real':'hexadec'}
            usecols = ['k', 'mono', 'quad', 'hexadec', 'modes']
            kwargs = {'usecols':usecols, 'mapcols':mapcols}
            poles = SpectraSet.from_files(loader, d, basename, [self.boxes], ['box'], args=('1d',), kwargs=kwargs, ignore_missing=True)

            # remove null
            valid_boxes = []
            for key, p in poles.nditer():
                if not p.isnull():
                    valid_boxes.append(key['box'].tolist())
            poles = poles.sel(box=valid_boxes)
        
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
            
            if subtract_shot_noise:
                for key in poles.ndindex():
                    if key['ell'] == 0:
                        p = poles.loc[key].values
                        p['power'] = p['power'] - p.attrs['shot_noise']
            
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
            
    def get_window(self, scaled=False):
        """
        Return the formatted window function for the cutsky challenge mocks
        """
        if scaled:
            filename = 'wilson_random_win_Ncutsky_0.4_fidcosmo_strim_7.00e+02_smooth_201x2.dat'
        else:
            filename = 'wilson_random_win_Ncutsky_0.3_truecosmo_strim_7.00e+02_smooth_201x2.dat'
        filename = os.path.join(self.root, 'extra', filename)
        return np.loadtxt(filename)