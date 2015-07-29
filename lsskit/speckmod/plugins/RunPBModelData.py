"""
    RunPBModelData.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugins for loading runPB data to be modeled
"""

from lsskit.speckmod.plugins import ModelInput
from lsskit import data as lss_data, numpy as np
from lsskit.speckmod import tools

from pyRSD import pygcl
import pandas as pd
import argparse

class SelectAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(SelectAction, self).__init__(option_strings, dest, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        fields = values.split(',')
        for f in fields:
            key, val = f.split('=')
            getattr(namespace, self.dest)[key.strip()] = eval(val.strip())
       
class RunPBModelData(object):
    """
    Base class to handle the input data from the runPB simulations
    """    
    def __init__(self, dict):
  
        # load the data and auxilliary info
        self.data = lss_data.PowerSpectraLoader.get('RunPB', self.path)
        self.biases = self.data.get_halo_biases()
        self.cosmo = pygcl.Cosmology("runPB.ini")
    
    def _to_dataframe(self, p):
        d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)
        return pd.DataFrame(data={'y':d['power'], 'error':d['error']}, index=pd.Index(d['k'], name='k'))
            
    def __iter__(self):
        """
        Iterate over the simulation data
        
        Returns
        -------
        (a, mass) : (str, int)
            the index, specified by the string `a` and int `mass`
        extra : dict
            any extra information, stored as a dictionary. contains keys
            `s8_z` for sigma8(z) and `b1` for the linear bias
        df : pandas.DataFrame
            dataframe holding the power as `y` column and error as `error` column
        """
        raise NotImplementedError
        
    @classmethod
    def register(cls):
        
        args = cls.name+":path"
        options = "[:-kmin=None][:-kmax=None][:-select='a=a_str, mass=mass_bin']"
        h = cls.add_parser(cls.name, usage=args+options)
        h.add_argument("path", type=str, help="the root directory of the data")
        h.add_argument('-kmin', type=float, help='the minimum wavenumber in h/Mpc to fit')
        h.add_argument('-kmax', type=float, help='the maximum wavenumber in h/Mpc to fit')
        h.add_argument('-select', action=SelectAction, help='the maximum wavenumber in h/Mpc to fit')
        h.set_defaults(klass=cls)
        
        
class PhhRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space halo-halo auto spectra data
    (shot-noise subtracted) from the runPB simulation across several redshifts 
    and mass bins.
    """
    name = 'PhhRunPBData'
    plugin_type = 'data'
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
        
    @property
    def _data(self):
        d = self.data.get_Phh('real')
        if self.select is not None:
            d = d.sel(**self.select)
        return d
        
    def __iter__(self):
        Phh = self._data
        extra = {}
        extra['cosmo'] = self.cosmo
        
        for key, val in Phh.nditer():
            
            # bias value for this mass bin
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
            
            # redshift and sigma8(z) value
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            extra['z'] = z
                    
            # make the dataframe and subtract shot noise
            df = self._to_dataframe(val)
            power = val.values
            df['y'] -= power.box_size**3 / power.N1

            # yield the index, extra dict, and power spectrum
            yield key, extra, df
            

class PhmResidualRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space halo-matter cross spectra data from
    the runPB simulation across several redshifts and mass bins. The 
    data that is returned is:
    
        :math: y = P_hm(k, z, M) - b_1 * Pzel(k, z),
    
    where Pzel(k,z) is the Zel'dovich matter power spectrum in real-space.
    """
    name = 'PhmResidualRunPBData'
    plugin_type = 'data'
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
    
    @property
    def _data(self):
        d = self.data.get_Phm('real')
        if self.select is not None:
            d = d.sel(**self.select)
        return d
    
    def __iter__(self):
        Phm = self._data
        if not hasattr(self, 'Pzel'):
            self.Pzel = pygcl.ZeldovichP00(self.cosmo, 0.)
            
        extra = {}
        extra['cosmo'] = self.cosmo
        for key, val in Phm.nditer():
                        
            # bias value for this mass bin
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
            
            # redshift and sigma8(z) value
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            extra['z'] = z
            
            # get the dataframe and subtract the Pzel term
            self.Pzel.SetRedshift(z)
            df = self._to_dataframe(val)
            df['y'] -= extra['b1']*self.Pzel(df.index.values)
            
            # yield the index, extra dict, and power spectrum
            yield key, extra, df
    

class LambdaARunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space, type A stochasticity data from
    the runPB simulation across several redshifts and mass bins. 
    """
    name = 'LambdaARunPBData'
    plugin_type = 'data'
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
    
    @property
    def _data(self):
        d = self.data.get_lambda(space='real', kind='A')
        if self.select is not None:
            d = d.sel(**self.select)
        return d
        
    def __iter__(self):
        lam = self._data      
        extra = {}
        extra['cosmo'] = self.cosmo
        for key, val in lam.nditer():
                        
            # bias value for this mass bin
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
            
            # redshift and sigma8(z) value
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            extra['z'] = z
            
            # yield the index, extra dict, and power spectrum
            yield key, extra, self._to_dataframe(val)  
        
class LambdaBRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space, type B stochasticity data from
    the runPB simulation across several redshifts and mass bins. 
    """
    name = 'LambdaBRunPBData'
    plugin_type = 'data'
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
    
    @property
    def _data(self):
        d = self.data.get_lambda(space='real', kind='B')
        if self.select is not None:
            d = d.sel(**self.select)
        return d
        
    def __iter__(self):
        lam = self._data  
        extra = {}
        extra['cosmo'] = self.cosmo
        for key, val in lam.nditer():
                        
            # bias value for this mass bin
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
            
            # redshift and sigma8(z) value
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            extra['z'] = z
            
            # yield the index, extra dict, and power spectrum
            yield key, extra, self._to_dataframe(val)          
    

class PhmRatioRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space halo-matter cross spectra data from
    the runPB simulation across several redshifts and mass bins. The 
    data that is returned is:
    
        :math: y = P_hm(k, z, M)  / b_1 * P_mm(k, z, M) - 1
    
    where Pzel(k,z) is the Zel'dovich matter power spectrum in real-space.
    """
    name = 'PhmRatioRunPBData'
    plugin_type = 'data'
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
    
    @property
    def _data(self):
        # Phm
        d1 = self.data.get_Phm('real')
        if self.select is not None:
            d1 = d1.sel(**self.select)
            
        # Pmm
        d2 = self.data.get_Pmm('real')
        if self.select is not None:
            key = {k:self.select[k] for k in self.select if k in d2.dims}
            d2 = d2.sel(**key)
            
        return d1, d2
    
    def _to_dataframe(self, Phm, Pmm, b1):
        Phm, Pmm = Phm.values, Pmm.values
        x = tools.get_valid_data(Phm, kmin=self.kmin, kmax=self.kmax)
        y = tools.get_valid_data(Pmm, kmin=self.kmin, kmax=self.kmax)
        y['power'] -= Pmm.box_size**3 / Pmm.N1
        
        sim_ratio = x['power']/(b1*y['power']) - 1.
        sim_err = (x['power']/y['power']/b1)*((y['error']/y['power'])**2 + (x['error']/x['power'])**2)**0.5
        
        return pd.DataFrame(data={'y':sim_ratio, 'error':sim_err}, index=pd.Index(x['k'], name='k'))
    
    def __iter__(self):
        
        Phm, Pmm = self._data
            
        extra = {}
        extra['cosmo'] = self.cosmo
        for key, x in Phm.nditer():
            
            # get this Pmm
            Pmm_key = {k:key[k] for k in key if k in Pmm.dims}
            y = Pmm.sel(**Pmm_key)
                        
            # bias value for this mass bin
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
            
            # redshift and sigma8(z) value
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            extra['z'] = z
                        
            # yield the index, extra dict, and power spectrum
            yield key, extra, self._to_dataframe(x, y, extra['b1'])
