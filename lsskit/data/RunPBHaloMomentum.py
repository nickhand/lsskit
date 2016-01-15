from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, io
from nbodykit.dataset import DataSet
        
import os

class RunPBHaloMomentum(PowerSpectraLoader):
    """
    Class to return the halo-related data for the RunPB simulations
    """
    name = "RunPBHaloMomentum"
    a    = ['0.5000', '0.5714', '0.6061', '0.6452', '0.6667', '0.6897', '0.7143', '0.8000', '0.9091', '1.0000']
    mass = range(8)
    mu   = [0, 2, 4, 6, 8]
    
    def __init__(self, root, realization='mean', dk=None):
        self.root = root
        self.tag = realization
        self.dk = dk
        
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls) 
        
    #--------------------------------------------------------------------------
    # internal functions
    #--------------------------------------------------------------------------
    def _get_moment(self, moments, los="", save_errors=False, name=None, sel_mu=None, ignore_cache=False):
        """
        Internal function to return any (ell, ell_prime) moment
        
        Parameters
        ----------
        moments : list of tuples
            list of tuples specifying the velocity moments to load
        los : str, {'x', 'y', 'z'} optional
            the string specifying the line-of-sight
        save_errors : bool, optional
            whether to pickle the errors for future loading
        name : str, optional
            use this string as the attribute name
        sel_mu : int, list of int, optional
            select specific only mu values
        ignore_cache : bool, optional
            if `True`, ignore any saved errors
        """
        if name is None:
            name = "_plus_".join("P%d%d" %(ell, ell_prime) for (ell, ell_prime) in moments)
        name_ = "_" + name
        if los: name_ += '_%slos' %los
        try:
            return getattr(self, name_)
        except AttributeError:

            coords = [self.a, self.mass]
            dims = ['a', 'mass']
            add_errors = True
            
            # load
            loader = io.load_momentum
            
            # loop over all moments
            toret = []
            for (ell, ell_prime) in moments:
                
                columns = None
                if ell == 0 and ell_prime == 0:
                    d = os.path.join(self.root, 'halo/real/density/power') 
                    tag = self.tag if 'mean' not in self.tag else '10mean'
                    basename = 'pk_hh{mass}_runPB_%s_{a}.dat' %tag
                    columns = ['k', 'power', 'modes']
                else:
                    d = os.path.join(self.root, 'halo/real/momentum/poles')
                    if los:
                        args = (ell, ell_prime, self.tag, los)
                        basename = 'poles_P%d%d_hh{mass}_runPB_%s_{a}_%slos.dat' %args
                    else:
                        args = (ell, ell_prime, self.tag)
                        basename = 'poles_P%d%d_hh{mass}_runPB_%s_{a}.dat' %args
                
                kwargs = {'sum_only':['modes'], 'force_index_match':True}
                if columns is not None: kwargs['columns'] = columns
                poles = SpectraSet.from_files(loader, d, basename, coords, dims, args=(ell,ell_prime,), kwargs=kwargs)
            
                # add the mu dimension
                data = np.asarray(poles.values.tolist())
                poles = SpectraSet(data, coords=coords+[self.mu], dims=dims+['mu'])
                
                # reindex 
                poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')
                                    
                # choose specific mu
                if sel_mu is not None:
                    poles = poles.sel(mu=sel_mu)
                
                # add errors
                if ell == 0 and ell_prime == 0:
                    poles.add_power_errors()
                    add_errors = False
                    
                toret.append(poles)
            
            if len(moments) != 0:
                for i in range(1, len(moments)):
                    for key, P in toret[i].nditer():
                        
                        subkey = {k:key[k] for k in key if k in toret[0].dims}
                        P0 = toret[0].sel(**subkey).values
                        P0['power'] += P.values['power']
            toret = toret[0]
            
            if add_errors:
                kw = {'sel_mu':sel_mu, 'save_errors':save_errors, 'name':name, 'ignore_cache':ignore_cache}
                self._add_errors(toret, moments, **kw)
                                
            setattr(self, name_, toret)
            return toret
            
    def _errors_from_realizations(self, a, mass, moments, sel_mu=None):
        """
        Internal function to compute the momentum moment errors as the 
        diagonal elements of the covariance matrix, which is constructed using 
        the 30 different realizations of each spectrum 
        (3 lines-of-sight + 10 realizations)
        
        Parameters
        ----------
        a : str
            the string specifying the scale factor
        mass : int
            the integer specifying the mass bin
        moments : list of tuples
            list of tuples specifying the velocity moments to load
        """
        from glob import glob
        from collections import defaultdict
    
        if isinstance(sel_mu, int): sel_mu = [sel_mu]
        if sel_mu is None: sel_mu = self.mu
        Nmu = len(self.mu)
        
        # the file pattern
        d = "/Volumes/Frodo/RSDResearch/RunPB/Results/halo/real/momentum/poles"
        data = defaultdict(list)
        
        for i, (ell, ell_prime) in enumerate(moments):
            
            args = (ell, ell_prime, mass, a)
            basename = 'poles_P%d%d_hh%d_runPB_*_%s_*los.dat' %args
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
                    if mu not in sel_mu: continue
                    if isinstance(loaded[k], DataSet):
                        if i == 0:
                            data[mu].append(loaded[k]['power'])
                        else:
                            data[mu][j] += loaded[k]['power']
                
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
            
    def _add_errors(self, poles, moments, save_errors=False, sel_mu=None, name=None, ignore_cache=False):
        """
        Internal function to add errors to the exisiting data sets, optionally
        loading or saving them to a pickle file
        """
        import pickle 
        
        if name is None:
            name = "_plus_".join("P%d%d" %(ell, ell_prime) for (ell, ell_prime) in moments)
        
        d = os.path.join(self.root, 'halo/real/momentum/poles/errors')
        reindex_tag = "" if self.dk is None else str(self.dk)+"_"
        args = (name, self.tag, reindex_tag)
        basename = 'poles_%s_hh{mass}_runPB_%s_{a}_%serrors.pickle' %args
       
        # compute errors for a, mass
        for key in poles.ndindex(dims=['a', 'mass']):
        
            error_file = os.path.join(d, basename.format(mass=key['mass'], a=key['a']))
            
            # load from pickle file
            if os.path.exists(error_file) and not ignore_cache:
                errors = pickle.load(open(error_file, 'r'))
            # compute fresh
            else:
                if not os.path.exists("/Volumes/Frodo"):
                    raise OSError("please mount `Frodo` in order to compute momentum errors")
                errors = self._errors_from_realizations(key['a'], key['mass'], moments, sel_mu=sel_mu)
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
    
    #--------------------------------------------------------------------------
    # momentum moments
    #--------------------------------------------------------------------------
    def get_P00(self):
        """
        Return real-space halo P00
        """
        return self._get_moment([(0, 0)])
            
    def get_P01(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real-space halo P01
        """
        return self._get_moment([(0, 1)], los=los, save_errors=save_errors, ignore_cache=ignore_cache)
        
    def get_P11(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real-space halo P11
        """
        return self._get_moment([(1, 1)], los=los, save_errors=save_errors, ignore_cache=ignore_cache)
        
    def get_P02(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real-space halo P02
        """
        return self._get_moment([(0, 2)], los=los, save_errors=save_errors, ignore_cache=ignore_cache)
                
    def get_P12(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real-space halo P12
        """
        return self._get_moment([(1, 2)], los=los, save_errors=save_errors, ignore_cache=ignore_cache)
        
    def get_P03(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real-space halo P03
        """
        return self._get_moment([(0, 3)], los=los, save_errors=save_errors, ignore_cache=ignore_cache)
        
    def get_P13(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real-space halo P13
        """
        return self._get_moment([(1, 3)], los=los, save_errors=save_errors, ignore_cache=ignore_cache)
        
    def get_P22(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real-space halo P22
        """
        return self._get_moment([(2, 2)], los=los, save_errors=save_errors, ignore_cache=ignore_cache)
        
    def get_P04(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real-space halo P04
        """
        return self._get_moment([(0, 4)], los=los, save_errors=save_errors, ignore_cache=ignore_cache)
        
    #--------------------------------------------------------------------------
    # sum of moments
    #--------------------------------------------------------------------------
    def get_P11_plus_P02(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real space halo P11 + P02
        """
        kw = {'los':los, 'save_errors':save_errors, 'ignore_cache':ignore_cache}
        moments = [(1, 1), (0, 2)]
        return self._get_moment(moments, **kw)
        
    def get_P12_plus_P03(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real space halo P12 + P03
        """
        kw = {'los':los, 'save_errors':save_errors, 'ignore_cache':ignore_cache}
        moments = [(1, 2), (0, 3)]
        return self._get_moment(moments, **kw)
        
    def get_P13_plus_P22_plus_P04(self, los="", save_errors=False, ignore_cache=False):
        """
        Return real space halo P13 + P22 + P04
        """
        kw = {'los':los, 'save_errors':save_errors, 'ignore_cache':ignore_cache}
        moments = [(1, 3), (2, 2), (0, 4)]
        return self._get_moment(moments, **kw)
        
    def get_Pmu2(self, los="", save_errors=False, ignore_cache=False):
        """
        Return moments contributing to the total mu^2, namely P01, P11, P02
        """
        kw = {'los':los, 'save_errors':save_errors, 'ignore_cache':ignore_cache, 'name':'Pmu2', 'sel_mu':2}
        moments = [(0, 1), (1, 1), (0, 2)]
        return self._get_moment(moments, **kw)
        
    def get_Pmu4(self, los="", save_errors=False, ignore_cache=False):
        """
        Return moments contributing to the total mu^4, namely 
        P11, P02, P12, P03, P13, P22, P04
        """
        kw = {'los':los, 'save_errors':save_errors, 'ignore_cache':ignore_cache, 'name':'Pmu4', 'sel_mu':4}
        moments = [(1, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 2), (0, 4)]
        return self._get_moment(moments, **kw)
        
    def get_Pmu6(self, los="", save_errors=False, ignore_cache=False):
        """
        Return moments contributing to the total mu^6, namely 
        P12, P03, P13, P22, P04
        """
        kw = {'los':los, 'save_errors':save_errors, 'ignore_cache':ignore_cache, 'name':'Pmu6', 'sel_mu':6}
        moments = [(1, 2), (0, 3), (1, 3), (2, 2), (0, 4)]
        return self._get_moment(moments, **kw)
        
    def get_Pmu8(self, los="", save_errors=False, ignore_cache=False):
        """
        Return moments contributing to the total mu^8, namely P13, P22, P04
        """
        kw = {'los':los, 'save_errors':save_errors, 'ignore_cache':ignore_cache, 'name':'Pmu8', 'sel_mu':8}
        moments = [(1, 3), (2, 2), (0, 4)]
        return self._get_moment(moments, **kw)
    
    
    def get_vel_disp(self, los=""):
        """
        Return the mean velocity dispersion 
        """
        import pandas as pd
        import xray
        
        f = os.path.join(self.root, 'meta/halo_mean_vel_powers.hdf')
        if not os.path.exists(f):
            raise ValueError("the file `%s` does not exist" %f)
        
        stats = pd.read_hdf(f, 'data')
        
        # compute sigma sq from the DataFrame
        if not los:
            levels = ['a', 'mass']
            sigma_sq = (stats['N']*stats['vpar_2']).sum(level=levels) / stats['N'].sum(level=levels)
        else:
            real = self.tag.split('PB')[-1]
            sigma_sq = stats['vpar_2'].xs(los, level='los').xs(real, level='realization')
            
        values = sigma_sq.reshape((len(self.a), len(self.mass)))**0.5
        toret = xray.DataArray(values, dims=['a', 'mass'], coords=[self.a, self.mass])
        
        return toret
        
            
            
            
        
        
        
        
        
        
            
    
