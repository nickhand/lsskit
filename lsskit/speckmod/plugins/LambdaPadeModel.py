from lsskit.speckmod.plugins import ModelInput
import lmfit

class LambdaPadeModel(lmfit.Model, ModelInput):

    name = "LambdaPadeModel"
    plugin_type = 'model'
    param_names = ['A0', 'A1', 'R']
    
    def __init__(self, dict):
        
        ModelInput.__init__(self, dict)
        super(LambdaPadeModel, self).__init__(self.__call__, 
                                            independent_vars=['k'])
    
    @classmethod
    def register(cls):
        h = cls.add_parser(cls.name, usage=cls.name)
        h.set_defaults(klass=cls)
    
    def __call__(self, k, **kwargs):
        
        # get the parameters
        A0 = kwargs['A0']
        A1 = kwargs['A1']
        R = kwargs['R']
        
        return (A0 + A1*(k*R)**2) / (1 + (k*R)**2)
        
        
    

