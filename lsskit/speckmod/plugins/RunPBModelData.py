"""
    RunPBModelData.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugins for loading runPB data to be modeled
"""

from lsskit.speckmod.plugins import ModelInput
from lsskit import data as lss_data
from lsskit.speckmod import tools

from pyRSD import pygcl
import pandas as pd
       
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
        
    def __iter__(self):
        Phh = self.data.get_Phh('real')
        extra = {}
        for key, val in Phh.nditer():
            
            # bias value for this mass bin
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values
            
            # redshift and sigma8(z) value
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            
            # make the dataframe and subtract shot noise
            df = self._to_dataframe(val)
            power = val.values
            df['y'] -= power.box_size**3 / power.N1
            
            # yield the index, extra dict, and power spectrum
            yield (key['a'], key['mass']), extra, df
            

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
    
    def __iter__(self):
        Phm = self.data.get_Phm('real')
        if not hasattr(self, 'Pzel'):
            self.Pzel = pygcl.ZeldovichP00(self.cosmo, 0.)
            
        extra = {}
        for key, val in Phm.nditer():
                        
            # bias value for this mass bin
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values
            
            # redshift and sigma8(z) value
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            
            # get the dataframe and subtract the Pzel term
            self.Pzel.SetRedshift(z)
            df = self._to_dataframe(val)
            df['y'] -= extra['b1']*self.Pzel(df.index.values)
            
            # yield the index, extra dict, and power spectrum
            yield (key['a'], key['mass']), extra, df
    

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
    
    def __iter__(self):
        lam = self.data.get_lambda(space='real', kind='A')            
        extra = {}
        for key, val in lam.nditer():
                        
            # bias value for this mass bin
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values
            
            # redshift and sigma8(z) value
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            
            # yield the index, extra dict, and power spectrum
            yield (key['a'], key['mass']), extra, self._to_dataframe(val)  
        
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
    
    def __iter__(self):
        lam = self.data.get_lambda(space='real', kind='B')            
        extra = {}
        for key, val in lam.nditer():
                        
            # bias value for this mass bin
            extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values
            
            # redshift and sigma8(z) value
            z = 1./float(key['a']) - 1.
            extra['s8_z'] = self.cosmo.Sigma8_z(z)
            
            # yield the index, extra dict, and power spectrum
            yield (key['a'], key['mass']), extra, self._to_dataframe(val)          
    

