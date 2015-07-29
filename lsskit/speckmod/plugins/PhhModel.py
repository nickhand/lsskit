from lsskit.speckmod.plugins import ModelInput
from pyRSD.rsd import power_halo
from pyRSD.rsd._cache import Cache, parameter, cached_property
import lmfit

class PhhModel(lmfit.Model, ModelInput, Cache):

    name = "PhhModel"
    plugin_type = 'model'
    param_names = ['A1', 'A2', 'dP', 'R']
    
    def __init__(self, dict):
        
        ModelInput.__init__(self, dict)
        Cache.__init__(self)
        super(PhhModel, self).__init__(self.__call__, 
                                            independent_vars=['k'])
    
    @classmethod
    def register(cls):
        h = cls.add_parser(cls.name, usage=cls.name)
        h.set_defaults(klass=cls)
    
    @parameter
    def k(self, val):
        return val
        
    @parameter
    def z(self, val):
        if hasattr(self, '_Phalo'): self._Phalo.z = val
        return val
            
    @parameter
    def cosmo(self, val):
        return val
    
    @cached_property('z', 'cosmo')        
    def _Phalo(self):
        
        kwargs = {}
        kwargs['z'] = self.z
        kwargs['cosmo'] = self.cosmo
        kwargs['include_2loop'] = False
        kwargs['transfer_fit'] = "CLASS"
        kwargs['max_mu'] = 0
        kwargs['sigmav_from_sims'] = True
        kwargs['interpolate'] = False
        kwargs['use_P00_model'] = True
        return power_halo.HaloSpectrum(**kwargs)
            
    @cached_property('k', '_Phalo')
    def _Phh_2halo(self):
        return self._Phalo.P00_model.zeldovich_power(self.k)
        
    @cached_property('k', '_Phalo')
    def _Phh_1halo(self):
        return self._Phalo.P00_model.broadband_power(self.k)
    
    def __call__(self, k, **kwargs):
        
        # get the parameters
        A1 = kwargs['A1']
        A2 = kwargs['A2']
        dP = kwargs['dP']
        R = kwargs['R']
        
        # set the appropriate variables
        self.cosmo = kwargs['cosmo'].GetParamFile()
        self.z = kwargs['z']
        self.k = k
        
        return A1**2 * self._Phh_2halo + A2 * self._Phh_1halo + dP/(1 + (k*R)**2)
        
        
    

