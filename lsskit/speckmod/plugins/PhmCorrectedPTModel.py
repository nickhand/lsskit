from lsskit.speckmod.plugins import ModelInput
from lsskit import numpy as np
from pyRSD.rsd import power_halo
from pyRSD.rsd._cache import Cache, parameter, cached_property
import lmfit

class PhmCorrectedPTModel(lmfit.Model, ModelInput, Cache):

    name = "PhmCorrectedPTModel"
    plugin_type = 'model'
    param_names = ['A0', 'A1', 'b2_00', 'k_t']
    
    def __init__(self, dict):
        
        ModelInput.__init__(self, dict)
        Cache.__init__(self)
        super(PhmCorrectedPTModel, self).__init__(self.__call__, 
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
    def _K00(self):
        return self._Phalo.K00(self.k)
        
    @cached_property('k', '_Phalo')
    def _P00(self):
        return self._Phalo.P00_model(self.k)
            
    def transition(self, k, k_transition=0.4, b=0.05):
        return 0.5 + 0.5*np.tanh((k-k_transition)/b)
    
    def __call__(self, k, **kwargs):
        
        # get the parameters
        A0 = kwargs['A0']
        A1 = kwargs['A1']
        b2_00 = kwargs['b2_00']
        k_t = kwargs.get('k_t', 0.4)
        
        # get the Phalo model
        self.cosmo = kwargs['cosmo'].GetParamFile()
        self.z = kwargs['z']
        self.k = k
        
        # computed normed K00
        b1 = kwargs['b1']
        normed_K00 = self._K00 / b1 / self._P00

        # return the model
        switch = self.transition(k, k_t)
        return (1-switch)*b2_00*normed_K00 + switch*(A1*k + A0)
        
        
    

