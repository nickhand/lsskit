from lsskit.speckmod.plugins import ModelInput
import lmfit
from lsskit import numpy as np

class LambdaLogModel(lmfit.Model, ModelInput):

    name = "LambdaLogModel"
    plugin_type = 'model'
    param_names = ['A0', 'A1']
    
    def __init__(self, dict):
        
        ModelInput.__init__(self, dict)
        super(LambdaLogModel, self).__init__(self.__call__, 
                                            independent_vars=['k'])
    
    @classmethod
    def register(cls):
        h = cls.add_parser(cls.name, usage=cls.name)
        h.set_defaults(klass=cls)
    
    def __call__(self, k, **kwargs):
        
        # get the parameters
        A0 = kwargs['A0']
        A1 = kwargs['A1']
        return A0 + A1*np.log(k)
        
    

