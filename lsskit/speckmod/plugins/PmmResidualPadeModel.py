from lsskit.speckmod.plugins import ModelInput
import lmfit
from pyRSD.rsd.halo_zeldovich import HaloZeldovichP00
from pyRSD.rsd._cache import Cache, parameter, cached_property

class PmmResidualPadeModel(lmfit.Model, ModelInput, Cache):

    name = "PmmResidualPadeModel"
    plugin_type = 'model'
    param_names = ['A0_amp', 'A0_slope', 'R1_amp', 'R1_slope', 'R1h_amp', 'R1h_slope', 'R2h_amp', 'R2h_slope']
    
    def __init__(self, dict):
        Cache.__init__(self)
        ModelInput.__init__(self, dict)
        super(PmmResidualPadeModel, self).__init__(self.__call__, 
                                            independent_vars=['k'])
    
    @classmethod
    def register(cls):
        h = cls.add_parser(cls.name, usage=cls.name)
        h.set_defaults(klass=cls)
    
    @parameter
    def z(self, val):
        if hasattr(self, '_Phz'): self._Phz.z = val
        return val
            
    @parameter
    def cosmo(self, val):
        return val
    
    @cached_property()        
    def _Phz(self):
        return HaloZeldovichP00(self.cosmo, 0., self.cosmo.sigma8(), True)
            
    def __call__(self, k, **kwargs):
        
        # set the appropriate variables
        self.cosmo = kwargs['cosmo']
        self.z = kwargs['z']
        
        # get the parameters
        self._Phz._A0_amp = kwargs['A0_amp']
        self._Phz._A0_slope = kwargs['A0_slope']
        self._Phz._R1_amp = kwargs['R1_amp']
        self._Phz._R1_slope = kwargs['R1_slope']
        self._Phz._R1h_amp = kwargs['R1h_amp']
        self._Phz._R1h_slope = kwargs['R1h_slope']
        self._Phz._R2h_amp = kwargs['R2h_amp']
        self._Phz._R2h_slope = kwargs['R2h_slope']
        
        return self._Phz.broadband_power(k)
        
        
    

