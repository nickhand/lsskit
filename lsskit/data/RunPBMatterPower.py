"""
    RunPBMatter.py

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : dark matter results for RunPB mocks
"""
from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, io
import os
        
class RunPBMatterPower(PowerSpectraLoader):
    """
    Class to load RunPB simulation measurements for dark matter
    at several redshifts
    """
    name = "RunPBMatterPower"
    a = ['0.5000', '0.5714', '0.6061', '0.6452', '0.6667', '0.6897', 
            '0.7143', '0.8000', '0.9091', '1.0000']
    
    def __init__(self, root, realization='mean', dk=None):
        
        self.root = root
        self.tag  = realization
        self.dk   = dk

    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls) 
    
    def get_Pmm(self, space='real'):
        """
        Return density auto-correlation in the space specified, 
        either `real` or `redshift`
        """
        name = '_Pmm_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            tag = self.tag
            columns = None
            d = os.path.join(self.root, 'matter', space, 'power')
            if space == 'real':
                basename = 'pk_mm_runPB_%s_{a}.dat' %tag
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_mm_runPB_%s_{a}_Nmu5.dat' %tag
                
            coords = [self.a]
            loader = io.load_power
            kwargs = {}
            if columns is not None: kwargs['columns'] = columns
            Pmm = SpectraSet.from_files(loader, d, basename, coords, dims=['a'], 
                                            args=('1d',), kwargs=kwargs)
            
            # reindex
            Pmm = self.reindex(Pmm, 'k_cen', self.dk, weights='modes')
            
            # add errors
            Pmm.add_power_errors()
            
            setattr(self, name, Pmm)
            return Pmm
            
            
    def get_Pdv(self):
        """
        Return density - velocity divergence cross power spectrum in
        real space
        """
        try:
            return self._Pdv
        except AttributeError:

            d = os.path.join(self.root, 'matter/real/poles')
            basename = 'poles_mm_Pdv_runPB_%s_{a}.dat' %self.tag

            coords = [self.a]
            loader = io.load_power
            kwargs = {'usecols':['k', 'power', 'modes'], 'mapcols':{'power_0.real':'power'}}
            
            Pdv = SpectraSet.from_files(loader, d, basename, coords, dims=['a'], 
                                            args=('1d',), kwargs=kwargs, ignore_missing=True)
            Pdv.add_errors(self.get_Pmm(space='real'), self.get_Pvv())
            self._Pdv = Pdv
            return Pdv

    def get_Pvv(self):
        """
        Return the velocity divergence auto power spectrum in
        real space
        """
        try:
            return self._Pvv
        except AttributeError:

            d = os.path.join(self.root, 'matter/real/poles')
            basename = 'poles_mm_Pvv_runPB_%s_{a}.dat' %self.tag

            coords = [self.a]
            loader = io.load_power
            kwargs = {'usecols':['k', 'power', 'modes'], 'mapcols':{'power_0.real':'power'}}
            
            Pvv = SpectraSet.from_files(loader, d, basename, coords, dims=['a'], 
                                            args=('1d',), kwargs=kwargs, ignore_missing=True)
            Pvv.add_errors()
            self._Pvv = Pvv
            return Pvv
    
    def get_P01(self, los=""):
        """
        Return either the `mu2` component of the real-space
        P01 density - radial velocity correlator
        """
        name = '_P01'
        if los: name += '_%slos' %los
        try:
            return getattr(self, name)
        except AttributeError:

            d = os.path.join(self.root, 'matter/real/poles')
            if los:
                basename = 'poles_mm_P01_runPB_%s_{a}_%slos.dat' %(self.tag, los)
            else:
                basename = 'poles_mm_P01_runPB_%s_{a}.dat' %(self.tag)
            coords = [self.a]
            dims = ['a']

            # load
            loader = io.load_momentum
            kwargs = {'sum_only':['modes'], 'force_index_match':True}
            poles = SpectraSet.from_files(loader, d, basename, coords, dims, args=(0,1,), kwargs=kwargs)

            # add the mu dimension
            data = np.asarray(poles.values.tolist())
            poles = SpectraSet(data, coords=[self.a, [2, 4, 6]], dims=['a', 'mu'])
            
            # reindex 
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')
                
            setattr(self, name, poles)
            return poles
    
    def get_P11(self, los=''):
        """
        Return either the `mu2` or `mu4` component of the real-space
        P11 velocity correlator
        """
        name = '_P11'
        if los: name += '_%slos' %los
        
        try:
            return getattr(self, name)
        except AttributeError:

            # try to load velocity shot noise
            import pandas as pd
            stats = pd.read_hdf(os.path.join(self.root, 'meta/matter_mean_sq_vel.hdf'), 'data')

            d = os.path.join(self.root, 'matter/real/poles')
            if los:
                basename = 'poles_mm_P11_runPB_%s_{a}_%slos.dat' %(self.tag, los)
            else:
                basename = 'poles_mm_P11_runPB_%s_{a}.dat' %(self.tag)
            coords = [self.a]
            dims = ['a']
            
            # load
            loader = io.load_momentum
            kwargs = {'sum_only':['modes'], 'force_index_match':True}
            poles = SpectraSet.from_files(loader, d, basename, coords, dims, args=(1,1,), kwargs=kwargs)
            
            # add the mu dimension
            data = np.asarray(poles.values.tolist())
            poles = SpectraSet(data, coords=[self.a, [2, 4, 6]], dims=['a', 'mu'])
            
            # reindex 
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')
            
            # not subtract the shot noise
            for key in poles.ndindex():
                
                # shot noise only in mu2 term
                if key['mu'] != 2:
                    continue
                    
                # compute the shot noise
                df = stats.loc[key['a']]
                if 'mean' in self.tag:
                    Pv2_shot = (df['vpar_2']*df['N']).sum() / df['N'].sum()
                else:
                    real = self.tag.split('PB')[-1]
                    Pv2_shot = df.loc[(los,real)]['vpar_2']
                    
                # get the pole
                pole = poles.sel(**key).values
                Pshot = pole.attrs['volume'] / pole.attrs['N1']  * Pv2_shot
                pole['power'] -= pole['k']**2 * Pshot
                pole.attrs['Pv2_shot'] = Pv2_shot

            setattr(self, name, poles)
            return poles
