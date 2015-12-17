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
    # multipoles data
    #--------------------------------------------------------------------------
    def get_mean_poles(self):
        """
        Return the mean of the cutsky galaxy multipoles in redshift space
        """
        try:
            return self._mean_poles
        except AttributeError:
        
            # read in the data
            d = os.path.join(self.root, 'data')
            basename = 'bianchips_cutsky_TSC_0.7_84mean.dat'
            f = os.path.join(d, basename)
            data = io.read_cutsky_power_poles(f, skiprows=0, sum_only=['modes'], force_index_match=True)
            
            # unstack the poles
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            data = tools.unstack_multipoles_one(data, ells, 'power')
            
            # make the SpectraSet and reindex
            poles = SpectraSet(data, coords=[[0, 2, 4]], dims=['ell'])
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')

            self._mean_poles = poles
            return poles

    def get_poles(self):
        """
        Return the cutsky galaxy multipoles in redshift space
        """
        name = '_poles'
        try:
            return self._poles           
        except AttributeError:
        
            # form the filename and load the data
            d = os.path.join(self.root, 'data')
            basename = 'bianchips_cutsky_TSC_0.7_{box:d}.dat'

            # read in the data
            loader = io.read_cutsky_power_poles
            kwargs = {'skiprows' : 31, 'sum_only':['modes'], 'force_index_match':True}
            poles = SpectraSet.from_files(loader, d, basename, [self.boxes], ['box'], kwargs=kwargs)
        
            # reindex
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')
            
            # unstack the poles
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            poles = tools.unstack_multipoles(poles, ells, 'power')
        
            self._poles = poles
            return poles