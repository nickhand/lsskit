from lsskit.speckmod.plugins import InputModel
import lmfit

def list_str(value):
    return value.split()
         
class PhmPadeModel(lmfit.Model, InputModel):

    name = "PhmPadeModel"
    param_names = ['A0', 'R', 'R1', 'R1h', 'R2h']
    
    def __init__(self, dict):
        
        InputModel.__init__(self, dict)
        super(lmfit.Model, self).__call__(self.__call__, 
                                            param_names=self.param_names, 
                                            independent_vars=self.indep_vars)
        for p in self.param_names:
            if p not in PhmPadeModel.param_names:
                raise ValueError("invalid parameter name `%s`; must be one of %s" %(p, PhmPadeModel.param_names))
    
    @classmethod
    def register(cls):
        
        usage = cls.name+":indep_vars:param_names"
        h = cls.add_parser(cls.name, usage=usage)
        h.add_argument("indep_vars", type=list_str, help="list of independent variable names")
        h.add_argument("param_names", type=list_str, help="list of parameter names")
        h.set_defaults(class=cls)
        
    def __call__(self, **kwargs):
        
        k = kwargs['k']
        A0 = kwargs['A0']
        R = kwargs['R']
        R1 = kwargs['R1']
        R1h = kwargs['R1h']
        R2h = kwargs['R2h']
        F = 1. - 1./(1. + (k*R)**2)
        return A0 * (1 + (k*R1)**2) / (1. + (k*R1h)**2 + (k*R2h)**4) * F
        
        
    

