from lsskit.speckmod.plugins import ModelInput
import lmfit

class PhmResidualPadeModel(lmfit.Model, ModelInput):

    name = "PhmResidualPadeModel"
    plugin_type = 'model'
    param_names = ['A0', 'R1', 'R1h', 'R2h']
    
    def __init__(self, dict):
        
        ModelInput.__init__(self, dict)
        super(PhmResidualPadeModel, self).__init__(self.__call__, 
                                            independent_vars=['k'])
        if self.fit_R:
            self.param_names += ['R']
    
    @classmethod
    def register(cls):
        h = cls.add_parser(cls.name, usage=cls.name)
        h.add_argument("--fit_R", help="whether or not to fit R parameter", action='store_true')
        h.set_defaults(klass=cls)
        
    def R_dm(self, s8_z):
        return 26. * (s8_z/0.8)**0.15
    
    def __call__(self, k, **kwargs):
        
        # get the parameters
        A0 = kwargs['A0']
        R1h = kwargs['R1h']
        
        # with defaults
        if 'R' in self.param_names:
            R = kwargs['R']
        else:
            R = self.R_dm(kwargs['s8_z'])
        R1 = kwargs['R1']
        R2h = kwargs['R2h']

        # now return
        F = 1. - 1./(1. + (k*R)**2)
        return A0 * (1 + (k*R1)**2) / (1. + (k*R1h)**2 + (k*R2h)**4) * F
        
class PhmResidualAllPadeModel(lmfit.Model, ModelInput):

    name = "PhmResidualAllPadeModel"
    plugin_type = 'model'
    param_names = ['A0', 'A0_dm', 'A0_slope', 'A0_dm_slope', 'R1', 'R1_zslope', 'R1_Mslope',\
                    'R1h', 'R1h_zslope', 'R1h_Mslope', 'R2h', 'R2h_zslope', 'R2h_Mslope',\
                    'R0', 'R0_dm', 'R0_slope', 'R0_dm_slope']
    
    def __init__(self, dict):
        
        ModelInput.__init__(self, dict)
        super(PhmResidualAllPadeModel, self).__init__(self.__call__, 
                                            independent_vars=['z', 's8_z', 'M', 'k'])
                                            
        self.radius_names = ['R1', 'R1h', 'R2h']
    
    @classmethod
    def register(cls):
        h = cls.add_parser(cls.name, usage=cls.name)
        h.set_defaults(klass=cls)
        
    def R_dm(self, s8_z):
        return 26. * (s8_z/0.8)**0.15
        
    def R0_model(self, s8_z, M, R, R_dm, alpha, beta):
        return R_dm*(s8_z/0.8)**beta - R*(M/1e13)**alpha
    
    def A0_model(self, s8_z, M, A, A_dm, alpha, beta, rhobar):
        return A*(M/rhobar)**alpha + A_dm*(s8_z/0.8)**beta
        
    def radius_model(self, M, z, A, alpha, beta):
        return A * (1+z)**alpha * (M/1e13)**beta
    
    def __call__(self, z, s8_z, M, k, **kwargs):
        
        Rs = {}
        for name in self.radius_names:
            A = kwargs[name]
            alpha = kwargs[name+"_zslope"]
            beta = kwargs[name+"_Mslope"]
            Rs[name] = self.radius_model(M, z, A, alpha, beta)
        R = self.R0_model(s8_z, M, kwargs['R0'], kwargs['R0_dm'], kwargs['R0_slope'], kwargs['R0_dm_slope'])
        A0 = self.A0_model(s8_z, M, kwargs['A0'], kwargs['A0_dm'], kwargs['A0_slope'], kwargs['A0_dm_slope'], kwargs['rho_bar'])

        # now return
        F = 1. - 1./(1. + (k*R)**2)
        return A0 * (1 + (k*Rs['R1'])**2) / (1. + (k*Rs['R1h'])**2 + (k*Rs['R2h'])**4) * F
        
        
    

