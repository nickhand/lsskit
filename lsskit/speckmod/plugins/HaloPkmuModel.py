from lsskit.speckmod.plugins import ModelInput
import lmfit

class HaloPkmuModel(lmfit.Model, ModelInput):

    name = "HaloPkmuModel"
    plugin_type = 'model'
    param_names = ['A']
    
    def __init__(self, dict):
        
        ModelInput.__init__(self, dict)
        super(HaloPkmuModel, self).__init__(self.__call__, 
                                                independent_vars=['k', 'mu'])
    
    @classmethod
    def register(cls):
        h = cls.add_parser(cls.name, usage=cls.name)
        h.set_defaults(klass=cls)
    
    def __call__(self, k, mu, **kwargs):
        
        # get the parameters
        A = kwargs['A']
        
        # set the appropriate variables
        Phalo = kwargs['Phalo']
        Phalo.z = kwargs['z']
        Phalo.f = Phalo.cosmo.f_z(Phalo.z)
        Phalo.sigma8_z = Phalo.cosmo.Sigma8_z(Phalo.z)
        Phalo.b1 = kwargs['b1']
        
        toret = Phalo.P_mu0(k) + mu**2 * Phalo.P_mu2(k) + mu**4 * Phalo.P_mu4(k) + A * mu**6 * Phalo.P_mu6(k)
        return toret
        
        
    

