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
  
        # load all the data and auxilliary info
        self.all_data = lss_data.PowerSpectraLoader.get('RunPB', self.path)
        self.biases = self.all_data.get_halo_biases()
        self.cosmo = pygcl.Cosmology("runPB.ini")
            
    def _make_dataframe(self, d):
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
        extra = {}
        extra['cosmo'] = self.cosmo
        d = self.data if not isinstance(self.data, (list, tuple)) else self.data[0]
        for key in d.ndindex():
            
            # add z, b1, and sigma8(z) to extra dict
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            extra['z'] = z
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()

            # yield the index, extra dict, and power spectrum
            yield key, extra, self.to_dataframe(key)
        
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
        
#------------------------------------------------------------------------------
class PhhRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space halo-halo auto spectra data
    (shot-noise subtracted) from the runPB simulation across several redshifts 
    and mass bins.
    """
    name = 'PhhRunPBData'
    plugin_type = 'data'
    variable_str = r"$P^{\ hh}(k)$"
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
        
    def to_dataframe(self, key):
        
        # power instance and shot noise
        p = self.data.sel(**key)
        Pshot = p.box_size**3 / p.N1
        
        # get the valid entries
        d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)
        d['power'] -= Pshot
        
        return self._make_dataframe(d)
        
    @property
    def data(self):
        d = self.all_data.get_Phh('real')
        if self.select is not None:
            d = d.sel(**self.select)
        return d

#------------------------------------------------------------------------------
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
    variable_str = r"$P^{\ hm} - b_1 \ P_\mathrm{zel}$"
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
        
    def to_dataframe(self, key):
        
        # make sure 
        if not hasattr(self, 'Pzel'):
            self.Pzel = pygcl.ZeldovichP00(self.cosmo, 0.)
            
        # power instance and shot noise
        p = self.data.sel(**key)
                
        # bias
        b1 = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
        
        # get the valid entries and subtract b1*Pzel
        d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)
        self.Pzel.SetRedshift(z)
        d['power'] -= b1*self.Pzel(d['k'])
                
        return self._make_dataframe(d)
    
    @property
    def data(self):
        d = self.all_data.get_Phm('real')
        if self.select is not None:
            d = d.sel(**self.select)
        return d
    
#------------------------------------------------------------------------------
class LambdaARunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space, type A stochasticity data from
    the runPB simulation across several redshifts and mass bins. 
    """
    name = 'LambdaARunPBData'
    plugin_type = 'data'
    variable_str = r"$\Lambda_A(k)$"
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
    
    def to_dataframe(self, key):
        p = self.data.sel(**key)
        d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)                
        return self._make_dataframe(d)
    
    @property
    def data(self):
        d = self.all_data.get_lambda(space='real', kind='A')
        if self.select is not None:
            d = d.sel(**self.select)
        return d
        
#------------------------------------------------------------------------------        
class LambdaBRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space, type B stochasticity data from
    the runPB simulation across several redshifts and mass bins. 
    """
    name = 'LambdaBRunPBData'
    plugin_type = 'data'
    variable_str = r"$\Lambda_B(k)$"
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
    
    def to_dataframe(self, key):
        key = {k:key[k] for k in key if k in self.data.dims}
        p = self.data.sel(**key)
        d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)                
        return self._make_dataframe(d)
        
    @property
    def data(self):
        d = self.all_data.get_lambda(space='real', kind='B')
        if self.select is not None:
            d = d.sel(**self.select)
        return d
        
#------------------------------------------------------------------------------
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
    variable_str = r"$P^{\ hm} / b_1 P^{\ mm} - 1$"
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
    
    def to_dataframe(self, key):
        # the data
        d = self.data
        
        # get Phm 
        Phm = d[0].sel(**key).values
        
        # get Pmm
        subkey = {k:key[k] for k in key if k in d[1].dims}
        Pmm = d[1].sel(**subkey).values
        
        # linear bias
        b1 = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
        
        # select valid data and subtract shot noise
        x = tools.get_valid_data(Phm, kmin=self.kmin, kmax=self.kmax)
        y = tools.get_valid_data(Pmm, kmin=self.kmin, kmax=self.kmax)
        y['power'] -= Pmm.box_size**3 / Pmm.N1
        
        sim_ratio = x['power']/(b1*y['power']) - 1.
        sim_err = (x['power']/y['power']/b1)*((y['error']/y['power'])**2 + (x['error']/x['power'])**2)**0.5
        
        return self._make_dataframe({'k':x['k'], 'power':sim_ratio, 'error':sim_err})
        
    @property
    def data(self):
        # Phm
        d1 = self.all_data.get_Phm('real')
        if self.select is not None:
            d1 = d1.sel(**self.select)
            
        # Pmm
        d2 = self.all_data.get_Pmm('real')
        if self.select is not None:
            key = {k:self.select[k] for k in self.select if k in d2.dims}
            d2 = d2.sel(**key)
            
        return d1, d2
    
#------------------------------------------------------------------------------
