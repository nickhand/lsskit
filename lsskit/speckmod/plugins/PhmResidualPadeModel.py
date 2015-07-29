from lsskit.speckmod.plugins import ModelInput
import lmfit

class PhmResidualPadeModel(lmfit.Model, ModelInput):

    name = "PhmResidualPadeModel"
    plugin_type = 'model'
    param_names = ['A0', 'R', 'R1', 'R1h', 'R2h']
    
    def __init__(self, dict):
        
        ModelInput.__init__(self, dict)
        super(PhmResidualPadeModel, self).__init__(self.__call__, 
                                            independent_vars=['k'])
    
    @classmethod
    def register(cls):
        h = cls.add_parser(cls.name, usage=cls.name)
        h.set_defaults(klass=cls)
        
    def R_dm(self, s8_z):
        return 26. * (s8_z/0.8)**0.15
    
    def __call__(self, k, b1=None, s8_z=None, **params):
        
        # get the parameters
        A0 = params['A0']
        R1h = params['R1h']
        
        # with defaults
        R = params.get('R', self.R_dm(s8_z))
        R1 = params.get('R1', 0.)
        R2h = params.get('R2h', 0.)
        
        # now return
        F = 1. - 1./(1. + (k*R)**2)
        return A0 * (1 + (k*R1)**2) / (1. + (k*R1h)**2 + (k*R2h)**4) * F
        
        
    

