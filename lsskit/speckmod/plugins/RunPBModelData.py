"""
    RunPBModelData.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugins for loading runPB data to be modeled
"""

from lsskit.speckmod.plugins import ModelInput
from lsskit import data as lss_data, numpy as np
from lsskit.specksis import tools

from pyRSD.rsd import power_halo
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
        self.all_data = lss_data.PowerSpectraLoader.get('RunPBHalo', self.path)
        self.biases = self.all_data.get_fof_halo_biases()
        self.masses = self.all_data.get_fof_halo_masses()
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
            if 'mass' in key:
                extra['b1'] = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
            extra['Phalo'] = getattr(self, 'Phalo', None)

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
        h.add_argument('-select', type=str, action=SelectAction, help='the maximum wavenumber in h/Mpc to fit')
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
        
        # get the power spectrum instance
        subkey = {k:key[k] for k in key if k in self.data.dims}
        p = self.data.sel(**subkey).values
        Pshot = tools.get_Pshot(p)
        
        # get the valid entries
        d = tools.get_valid_data(p, kmin=self.kmin, kmax=self.kmax)
        d['power'] -= Pshot
        
        return self._make_dataframe(d)
        
    @property
    def data(self):
        d = self.all_data.get_fof_Phh('real')
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
            
        # get the power spectrum instance
        subkey = {k:key[k] for k in key if k in self.data.dims}
        p = self.data.sel(**subkey)
                
        # bias
        b1 = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
        
        # get the valid entries and subtract b1*Pzel
        d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)
        z = 1./float(key['a']) - 1.
        self.Pzel.SetRedshift(z)
        d['power'] -= b1*self.Pzel(d['k'])
                
        return self._make_dataframe(d)
    
    @property
    def data(self):
        d = self.all_data.get_fof_Phm('real')
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
        key = {k:key[k] for k in key if k in self.data.dims}
        p = self.data.sel(**key)    
        d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)                
        return self._make_dataframe(d)
    
    @property
    def data(self):
        d = self.all_data.get_fof_lambda(space='real', kind='A')
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
        d = self.all_data.get_fof_lambda(space='real', kind='B')
        if self.select is not None:
            d = d.sel(**self.select)
        return d
        
#------------------------------------------------------------------------------        
class LambdaBCrossRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space, type B cross stochasticity data from
    the runPB simulation across several redshifts and mass bins. 
    """
    name = 'LambdaBCrossRunPBData'
    plugin_type = 'data'
    variable_str = r"$\Lambda_B^\mathrm{cross}(k)$"
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
    
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
            extra['b1_1'] = self.biases.sel(a=key['a'], mass=key['mass1']).values.tolist()
            extra['b1_2'] = self.biases.sel(a=key['a'], mass=key['mass2']).values.tolist()
            extra['b1'] = (extra['b1_1']*extra['b1_2'])**0.5

            # yield the index, extra dict, and power spectrum
            yield key, extra, self.to_dataframe(key)
            
    def to_dataframe(self, key):
        key = {k:key[k] for k in key if k in self.data.dims}
        p = self.data.sel(**key)
        d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)                
        return self._make_dataframe(d)
        
    @property
    def data(self):
        d = self.all_data.get_fof_lambda_cross(space='real', kind='B')
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
        subkey = {k:key[k] for k in key if k in d[0].dims}
        Phm = d[0].sel(**subkey).values
        
        # get Pmm
        subkey = {k:key[k] for k in key if k in d[1].dims}
        Pmm = d[1].sel(**subkey).values
        
        # linear bias
        b1 = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
        
        # select valid data and subtract shot noise
        x = tools.get_valid_data(Phm, kmin=self.kmin, kmax=self.kmax)
        y = tools.get_valid_data(Pmm, kmin=self.kmin, kmax=self.kmax)
        y['power'] -= tools.get_Pshot(Pmm)
        
        sim_ratio = x['power']/(b1*y['power']) - 1.
        sim_err = (x['power']/y['power']/b1)*((y['error']/y['power'])**2 + (x['error']/x['power'])**2)**0.5
        
        return self._make_dataframe({'k':x['k'], 'power':sim_ratio, 'error':sim_err})
        
    @property
    def data(self):
        # Phm
        d1 = self.all_data.get_fof_Phm('real')
        if self.select is not None:
            d1 = d1.sel(**self.select)
            
        # Pmm
        d2 = self.all_data.get_Pmm('real')
        if self.select is not None:
            key = {k:self.select[k] for k in self.select if k in d2.dims}
            d2 = d2.sel(**key)
            
        return d1, d2
    
#------------------------------------------------------------------------------
class PmmResidualRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space matter auto spectra data from
    the runPB simulation across several redshifts and mass bins. The 
    data that is returned is:
    
        :math: y = P_mm(k, z, M) - Pzel(k, z),
    
    where Pzel(k,z) is the Zel'dovich matter power spectrum in real-space.
    """
    name = 'PmmResidualRunPBData'
    plugin_type = 'data'
    variable_str = r"$P^{\ mm} - P_\mathrm{zel}$"
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
        
    def to_dataframe(self, key):
        
        # make sure 
        if not hasattr(self, 'Pzel'):
            self.Pzel = pygcl.ZeldovichP00(self.cosmo, 0.)
            
        # get the power spectrum instance
        subkey = {k:key[k] for k in key if k in self.data.dims}
        p = self.data.sel(**subkey)

        # get the valid entries and subtract Pzel
        d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)
        z = 1./float(key['a']) - 1.
        self.Pzel.SetRedshift(z)
        d['power'] -= self.Pzel(d['k'])
                
        return self._make_dataframe(d)
    
    @property
    def data(self):
        d = self.all_data.get_Pmm('real')
        if self.select is not None:
            d = d.sel(**self.select)
        return d
    
#------------------------------------------------------------------------------    
class HaloPkmuRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the redshift-space halo-halo auto spectra data
    (shot-noise subtracted) from the runPB simulation across several redshifts 
    and mass bins.
    """    
    name = 'HaloPkmuRunPBData'
    plugin_type = 'data'
    variable_str = r"$P^{\ hh}(k, \mu)$"
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
             
    @property
    def Phalo(self):
        try:
            return self._Phalo
        except:
            kwargs = {}
            kwargs['z'] = 0.55
            kwargs['cosmo_filename'] = self.cosmo.GetParamFile()
            kwargs['include_2loop'] = False
            kwargs['transfer_fit'] = "CLASS"
            kwargs['sigmav_from_sims'] = False
            kwargs['use_mean_bias'] = False
            kwargs['use_tidal_bias'] = False
            kwargs['use_P00_model'] = True
            kwargs['use_P01_model'] = True
            kwargs['use_P11_model'] = True
            kwargs['use_Pdv_model'] = True
            kwargs['Phm_model'] = 'halo_zeldovich'
            kwargs['use_mu_corrections'] = False
            kwargs['interpolate'] = True
            kwargs['max_mu'] = 6
            self._Phalo = power_halo.HaloSpectrum(**kwargs)
            return self._Phalo
                    
    def to_dataframe(self, key):
        
        # get the power spectrum instance
        subkey = {k:key[k] for k in key if k in self.data.dims}
        p = self.data.sel(**subkey).values
        Pshot = tools.get_Pshot(p)
        
        # get the valid entries
        d = tools.get_valid_data(p, kmin=self.kmin, kmax=self.kmax)
        d['power'] -= Pshot
        d = d.ravel(order='F')
        
        return self._make_dataframe(d)
        
    def _make_dataframe(self, d):
        df = pd.DataFrame(data={'y':d['power'], 'error':d['error'], 'k':d['k'], 'mu':d['mu']})
        return df.set_index(['k', 'mu'])        
        
    @property
    def data(self):
        d = self.all_data.get_fof_Phh('redshift')
        if self.select is not None:
            d = d.sel(**self.select)
        return d


#------------------------------------------------------------------------------
class PhmResidualAllRunPBData(ModelInput, RunPBModelData):
    """
    A plugin to return the real-space halo-matter cross spectra data from
    the runPB simulation across several redshifts and mass bins. The 
    data that is returned is:
    
        :math: y = P_hm(k, z, M) - b_1 * Pzel(k, z),
    
    where Pzel(k,z) is the Zel'dovich matter power spectrum in real-space.
    """
    name = 'PhmResidualAllRunPBData'
    plugin_type = 'data'
    variable_str = r"$P^{\ hm} - b_1 \ P_\mathrm{zel}$"
    
    def __init__(self, dict):
        ModelInput.__init__(self, dict)
        RunPBModelData.__init__(self, dict)
        
    def to_dataframe(self, *args, **kwargs):
        
        d = self.data
        out = pd.DataFrame()
        
        # make sure 
        if not hasattr(self, 'Pzel'):
            self.Pzel = pygcl.ZeldovichP00(self.cosmo, 0.)
        
        for key in self.data.ndindex():
            
            # add z, b1, and sigma8(z) to extra dict
            z = 1./float(key['a']) - 1.
            s8_z = self.cosmo.Sigma8_z(z)
            b1 = self.biases.sel(a=key['a'], mass=key['mass']).values.tolist()
            M = self.masses.sel(a=key['a'], mass=key['mass']).values.tolist()
            
            # get the power spectrum instance
            subkey = {k:key[k] for k in key if k in self.data.dims}
            p = self.data.sel(**subkey)
            
            # valid data
            d = tools.get_valid_data(p.values, kmin=self.kmin, kmax=self.kmax)
            self.Pzel.SetRedshift(z)
            
            # make the tmp DF
            tmp = pd.DataFrame()
            tmp['k'] = d['k']
            tmp['y'] = d['power'] - b1*self.Pzel(d['k'])
            tmp['error'] = d['error']
            tmp['s8_z'] = np.repeat(s8_z, len(d['k']))
            tmp['M'] = np.repeat(M, len(d['k']))
            tmp['z'] = np.repeat(z, len(d['k']))
            out = out.append(tmp)
                       
        return out.set_index(['z', 's8_z', 'M', 'k'])    
    
    @property
    def data(self):
        d = self.all_data.get_fof_Phm('real')
        return d
        
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
        extra['rho_bar'] = self.cosmo.rho_bar_z(0.)
        yield None, extra, self.to_dataframe()
    

