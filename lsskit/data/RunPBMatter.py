"""
    RunPBMatter.py

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : dark matter results for RunPB mocks
"""
from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, utils, tools
import os
from nbodykit import pkresult, files

class RunPBMatter(PowerSpectraLoader):
    name = "RunPBMatter"
    a = ['0.5000', '0.5714', '0.6061', '0.6452', '0.6667', '0.6897', '0.7143', '0.8000', '0.9091', '1.0000']
    
    def __init__(self, root, realization='mean'):
        self.root = root
        self.tag = realization

    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls) 
    
    def get_Pmm(self, space='real'):
        """
        Return Pmm in the space specified, either `real` or `redshift`
        """
        try:
            return getattr(self, '_Pmm_'+space)
        except AttributeError:
            
            tag = self.tag
            if tag == 'mean': tag = '10'+tag
            
            d = os.path.join(self.root, 'matter', space, 'power')
            if space == 'real':
                basename = 'pk_mm_runPB_%s_{a}.dat' %tag
            else:
                basename = 'pkmu_mm_runPB_%s_{a}_Nmu5.dat' %tag
                
            coords = [self.a]
            Pmm = SpectraSet.from_files(d, basename, coords, ['a'])
            Pmm.add_errors()
            setattr(self, '_Pmm_'+space, Pmm)
            return Pmm 
        
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
            
            # read in the data
            data = []
            for i, f in utils.enum_files(d, basename, dims, coords, ignore_missing=True):
                a_str = self.a[i[0]]
                if f is not None:
                    d, meta = files.ReadPower1DPlainText(f)
                    
                    # grab the imag part of dipole x k
                    cols = meta.pop('cols')
                    k = d[:,0]; modes = d[:,-1]
                    dipole = d[:,cols.index('power_1.imag')]
                    power = k*dipole
                    
                    # cross error
                    error1 = (4./5)**0.5 * dipole
                    
                    # auto error
                    a = self.get_P11('mu4').sel(a=a_str).values
                    b = self.get_P11('mu2').sel(a=a_str).values
                    c = self.get_Pmm().sel(a=a_str).values
                    
                    Pshot = b.volume / b.N1  * b.P11_shot
                    P11_mu2 = b['power'].data + Pshot
                    P11_mu4 = a['power'].data
                    Pmm = c['power'].data
                    
                    error2 = (4./3*P11_mu2*Pmm + 4./5.*P11_mu4*Pmm)**0.5
                    error = k * 2**0.5 * (1./modes)**0.5 * (error1 + error2)
                    
                    d = np.vstack([k, power, error, modes]).T
                    pk = pkresult.PkResult.from_dict(d, ['k', 'power', 'error', 'modes'], **meta)
                    data.append(pk)
                else:
                    data.append(np.nan)
            result = SpectraSet(data, coords=coords, dims=dims)
            setattr(self, name, result)
            return result
    
    def get_P11(self, mu, los=''):
        """
        Return either the `mu2` or `mu4` component of the real-space
        P11 velocity correlator
        """
        if mu not in ['mu2', 'mu4']:
            raise ValueError("`mu` must be either 'mu2' or 'mu4'")
            
        name = '_P11_%s' %mu
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
            
            # read in the data
            data = []
            for i, f in utils.enum_files(d, basename, dims, coords, ignore_missing=True):
                
                # compute the shot noise
                df = stats.loc[self.a[i[0]]]
                if 'mean' in self.tag:
                    Pv2_shot = (df['vpar_2']*df['N']).sum() / df['N'].sum()
                else:
                    real = self.tag.split('PB')[-1]
                    Pv2_shot = df.loc[(los,real)]['vpar_2']
                
                if f is not None:
                    d, meta = files.ReadPower1DPlainText(f)
                    Pshot = meta['volume'] / meta['N1']  * Pv2_shot
                    
                    # need linear combo of mono + quad
                    cols = meta.pop('cols')
                    k = d[:,0]; modes = d[:,-1]
                    mono = d[:,cols.index('power_0.real')]
                    quad = d[:,cols.index('power_2.real')]
                    
                    P11_mu4 = 1.5 * quad
                    P11_mu2 = ((mono - Pshot) - 0.5 * quad)
                    
                    # quad error
                    sigma_sq = 5. * (P11_mu2**2 + 2*P11_mu2*Pshot + Pshot**2)
                    sigma_sq += 2.62 * (2*P11_mu2*P11_mu4 + 2*P11_mu4*Pshot)
                    sigma_sq += 2.14 * P11_mu4**2
                    quad_error = 2**0.5 * (2./modes)**0.5 * sigma_sq**0.5 # factor of two needed?
                    
                    # 
                    if mu == 'mu4':
                        power = k**2 * P11_mu4
                        error = 1.5 * k**2 * quad_error
                    else:
                        power = k**2 * P11_mu2
                        meta['P11_shot'] = Pv2_shot
                        
                        # mono error
                        sigma_sq = (P11_mu2**2 + 2*P11_mu2*Pshot + Pshot**2)
                        sigma_sq += 1./3 * (2*P11_mu2*P11_mu4 + 2*P11_mu4*Pshot)
                        sigma_sq += 1./5 * P11_mu4**2
                        error = 2**0.5 * (2./modes)**0.5 * sigma_sq**0.5
                        error = ( (k**2 * error)**2 + (0.5 * k**2 * quad_error)**2 )**0.5
                    
                    d = np.vstack([k, power, error, modes]).T
                    pk = pkresult.PkResult.from_dict(d, ['k', 'power', 'error', 'modes'], **meta)
                    data.append(pk)
                else:
                    data.append(np.nan)
            result = SpectraSet(data, coords=coords, dims=dims)
            setattr(self, name, result)
            return result
            