"""
    RunPBMatter.py

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : dark matter results for RunPB mocks
"""
from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, io
from nbodykit.dataset import DataSet
import os

        
class RunPBMatterPower(PowerSpectraLoader):
    """
    Class to load RunPB simulation measurements for dark matter
    at several redshifts
    """
    name = "RunPBMatterPower"
    a = ['0.5000', '0.5714', '0.6061', '0.6452', '0.6667', '0.6897', 
            '0.7143', '0.8000', '0.9091', '1.0000']
    mu = [0, 2, 4, 6, 8]
    
    def __init__(self, root, realization='mean', dk=None):
        
        self.root = root
        self.tag  = realization
        self.dk   = dk

    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls) 
    
    def _errors_from_realizations(self, a, ell, ell_prime):
        """
        Internal function to compute the momentum moment errors as the 
        diagonal elements of the covariance matrix, which is constructed using 
        the 30 different realizations of each spectrum 
        (3 lines-of-sight + 10 realizations)
        """
        from glob import glob
        from collections import defaultdict
    
        Nmu = len(self.mu)
        
        # the file pattern
        d = "/Volumes/Frodo/RSDResearch/RunPB/Results/matter/fourier/real/momentum/poles"
        data = defaultdict(list)
        
        args = (ell, ell_prime, a)
        basename = 'poles_P%d%d_runPB_*_%s_*los.dat' %args
        pattern = os.path.join(d, basename)
    
        # the matching files
        files = glob(pattern)
        kw = {'sum_only':['modes'], 'force_index_match':True}
    
        for j, f in enumerate(files):
        
            loaded = io.load_momentum(f, ell, ell_prime, **kw)
            if self.dk is not None:
                for ii, l in enumerate(loaded):
                    if isinstance(l, DataSet):
                        loaded[ii] = l.reindex('k_cen', self.dk, weights='modes')
        
            for k, mu in enumerate(self.mu):
                if isinstance(loaded[k], DataSet):
                    data[mu].append(loaded[k]['power'])

                
        toret = [None]*Nmu
        for i, mu in enumerate(self.mu):
            
            if len(data[mu]):
                cov = np.cov(np.nan_to_num(data[mu]), rowvar=False)
                errs = cov.diagonal()**0.5
        
                # divide by root N if mean
                if 'mean' in self.tag:
                    errs /= len(files)**0.5
                toret[i] = errs
        
        return toret
            
    def _add_errors(self, poles, ell, ell_prime, save_errors=False, ignore_cache=False):
        """
        Internal function to add errors to the exisiting data sets, optionally
        loading or saving them to a pickle file
        """
        import pickle 
        
        name = "P%d%d" %(ell, ell_prime)
        d = os.path.join(self.root, 'matter/fourier/real/momentum/poles/errors')
        reindex_tag = "" if self.dk is None else str(self.dk)+"_"
        args = (name, self.tag, reindex_tag)
        basename = 'poles_%s_runPB_%s_{a}_%serrors.pickle' %args
       
        # compute errors for a, mass
        for key in poles.ndindex(dims='a'):
        
            error_file = os.path.join(d, basename.format(a=key['a']))
            
            # load from pickle file
            if os.path.exists(error_file) and not ignore_cache:
                errors = pickle.load(open(error_file, 'r'))
            # compute fresh
            else:
                if not os.path.exists("/Volumes/Frodo"):
                    raise OSError("please mount `Frodo` in order to compute momentum errors")
                errors = self._errors_from_realizations(key['a'], ell, ell_prime)
                if save_errors:
                    pickle.dump(errors, open(error_file, 'w'))
            
            # do each mu
            subkey = key.copy()
            
            for i, mu in enumerate(self.mu):                
                if errors[i] is not None:
                    if 'mu' not in poles.dims:
                        subkey.pop('mu', None)
                    else:
                        subkey['mu'] = mu
                                       
                    P = poles.sel(**subkey).values
                    P['error'] = errors[i]
                    
        return poles
        
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
            if 'mean' in self.tag:
                tag = '10mean'
            
            columns = None
            d = os.path.join(self.root, 'matter/fourier', space, 'density/power')
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
            
    
    def get_P01(self, los="", save_errors=False, ignore_cache=False):
        """
        Return either the `mu2` component of the real-space
        P01 density - radial velocity correlator
        """
        name = '_P01'
        if los: name += '_%slos' %los
        try:
            return getattr(self, name)
        except AttributeError:

            d = os.path.join(self.root, 'matter/fourier/real/momentum/poles')
            if los:
                basename = 'poles_P01_runPB_%s_{a}_%slos.dat' %(self.tag, los)
            else:
                basename = 'poles_P01_runPB_%s_{a}.dat' %(self.tag)
            coords = [self.a]
            dims = ['a']

            # load
            loader = io.load_momentum
            kwargs = {'sum_only':['modes'], 'force_index_match':True}
            poles = SpectraSet.from_files(loader, d, basename, coords, dims, args=(0,1,), kwargs=kwargs)

            # add the mu dimension
            data = np.asarray(poles.values.tolist())
            poles = SpectraSet(data, coords=[self.a, [0, 2, 4, 6, 8]], dims=['a', 'mu'])
            
            # reindex 
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')
                
            # add errors
            kw = {'save_errors':save_errors, 'ignore_cache':ignore_cache}
            self._add_errors(poles, 0, 1, **kw)
            
            setattr(self, name, poles)
            return poles
    
    def get_P11(self, los='', save_errors=False, ignore_cache=False):
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

            d = os.path.join(self.root, 'matter/fourier/real/momentum/poles')
            if los:
                basename = 'poles_P11_runPB_%s_{a}_%slos.dat' %(self.tag, los)
            else:
                basename = 'poles_P11_runPB_%s_{a}.dat' %(self.tag)
            coords = [self.a]
            dims = ['a']
            
            # load
            loader = io.load_momentum
            kwargs = {'sum_only':['modes'], 'force_index_match':True}
            poles = SpectraSet.from_files(loader, d, basename, coords, dims, args=(1,1,), kwargs=kwargs)
            
            # add the mu dimension
            data = np.asarray(poles.values.tolist())
            poles = SpectraSet(data, coords=[self.a, [0, 2, 4, 6, 8]], dims=['a', 'mu'])
            
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

            # add errors
            kw = {'save_errors':save_errors, 'ignore_cache':ignore_cache}
            self._add_errors(poles, 1, 1, **kw)
            
            setattr(self, name, poles)
            return poles
