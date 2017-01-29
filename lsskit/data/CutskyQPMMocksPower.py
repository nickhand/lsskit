import os

from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, covariance, io, tools
    
class CutskyQPMMocksPower(PowerSpectraLoader):
    name = "CutskyQPMMocksPower"
    boxes = range(1, 991)
    
    def __init__(self, root, dk=None):
        self.root = root
        self.dk = dk
        
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
        
    #--------------------------------------------------------------------------
    # multipoles data
    #--------------------------------------------------------------------------
    def get_poles(self, average=None):
        """
        Return the cutsky galaxy multipoles in redshift space
        """
        if average is not None:
            if isinstance(average, str):
                average = [average]
        else:
            average = []
        
        name = '_poles'
        if len(average):
            name += '_' + '_'.join(average)
            
        try:
            return getattr(self, name)
        except AttributeError:
        
            # form the filename and load the data
            basename = 'bianchips_qpmdr12_TSC_{box:04d}.dat'
            
            # read in the data
            loader = io.read_cutsky_power_poles
            kwargs = {'skiprows' : 31, 'sum_only':['modes'], 'force_index_match':True}
            poles = SpectraSet.from_files(loader, self.root, basename, [self.boxes], ['box'], kwargs=kwargs)
        
            # reindex
            poles = self.reindex(poles, 'k', self.dk, weights='modes')
                        
            # unstack the poles
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            poles = tools.unstack_multipoles(poles, ells, 'power')
        
            if len(average):
                poles = poles.average(axis=average, weights='modes')
        
            setattr(self, name, poles)
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
        poles = self.get_poles()

        kwargs['extras'] = False
        return covariance.compute_pole_covariance(poles, [0, 2, 4], **kwargs)
            