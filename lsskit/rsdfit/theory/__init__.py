from .. import AttrDict, os
import copy
import numpy as np

def replace_vars(s, D):
    """
    Given a string s and a dictionary of variables D, replace all variable
    names with the value from the dict D

    variable names in s are denoted by '$(' at the beginning and ')' at
    the end, or '$' at the beginning and a non-variable character at the
    end.  Variables must be valid python variable names, that is they
    consist of only alphanumeric characters (A-Z,a-z,0-9) and underscores,
    and cannot start with a number.

    example:
    >> D = {'my_var1' : 'abc',
            'my_var2' : '123' }
    >> s = "I know my $(my_var1)s and $my_var2's"
    >> print replace_vars(s,D)

    I know my abcs and 123's
    """
    s_in = str(s)
    s_out = ''

    while True:
        i = s_in.find('$')
        if i==-1:
            s_out += s_in
            break

        s_out += s_in[:i]
        s_in = s_in[i+1:]

        if len(s_in)==0:
            raise ValueError("trailing $")

        elif s_in[0] == '(':
            i = s_in.find(')')
            if i==-1:
                raise ValueError("unmatched '('")
            var = s_in[1:i]

            s_in = s_in[i+1:]
            try:
                s_out += str(D[var])
            except:
                s_out += os.environ[var]

        else:
            var = ''
            i = 0
            while True:
                if i>=len(s_in):
                    break
                s = s_in[i]
                if (s >= 'a' and s <= 'z') \
                        or (s >= 'A' and s <= 'Z') \
                        or (s >= '0' and s <= '9') \
                        or s=='_':
                    var += s
                    i += 1
                else:
                    break
            s_in = s_in[i:]
            s_out += str(D[var])
    return s_out

def load_nbar(val):
    """
    Load the nbar from the shot noise file
    """
    import pickle    
    val = val.split(":")
    if '$' in val[0]:
        val[0] = replace_vars(val[0], {})
        
    with open(val[0], 'rb') as ff:
        nbar = pickle.load(ff, encoding='latin1')
    if len(val) > 1:
        sl = dict(eval(val[1]))
        
        index = ()
        for dim in nbar['dims']:
            key = nbar['coords'][dim]['data'].index(sl[dim])
            index += (key,)
            
        nbar = np.array(nbar['data'])[index]
    else:
        nbar = nbar['data']    
    
    nbar = 1.0 / nbar
    return nbar
     

class NBarParam(AttrDict):
    """
    Nbar can be loaded from a shot noise file
    """
    def __setattr__(self, key, val): 
        if key == 'fiducial':
            if isinstance(val, str):
                try:
                    val = load_nbar(val)
                except Exception as e:
                    msg = "error trying to load nbar from file: '%s'\n" %val
                    msg += '\toriginal exception: %s' %str(e)
                    raise ValueError(msg)
            
        return dict.__setattr__(self, key, val)

       
class ModelParams(AttrDict):
    
    defaults = {'z' : None, 
                'cosmo_filename' : None,
                'include_2loop' : False,
                'transfer_fit' : 'CLASS', 
                'vel_disp_from_sims' : False,
                'use_mean_bias' : False,
                'fog_model' : 'modified_lorentzian', 
                'use_tidal_bias' : False,
                'use_P00_model' : True,
                'use_P01_model' : True,
                'use_P11_model' : True, 
                'use_Phm_model' : True,
                'use_Pdv_model' : True,
                'Pdv_model_type' : 'jennings',
                'correct_mu2' : False,
                'correct_mu4' : False,
                'max_mu' : 4,
                'interpolate' : True,
                'use_so_correction' : False, 
                'enhance_wiggles' : False, 
                'use_vlah_biasing' : True}
                
    def __init__(self, *args, **kwargs):
        for name in self.defaults:
            self.setdefault(name, self.defaults[name])
        super(ModelParams, self).__init__(*args, **kwargs)
        
    def __str__(self):
        return "\n".join(["model.%s = %s" %(k,repr(v)) for k,v in sorted(self.items())])
            
class TheoryParams(object):
    """
    A base class to store the most general version of fitting parameters for the RSD model
    """    
    def __init__(self):
        cosmo        = ['sigma8_z', 'f', 'alpha_perp', 'alpha_par']
        biases       = ['b1_cA', 'b1_cB', 'b1_sA', 'b1_sB']
        fractions    = ['fcB', 'fsB', 'fs', 'Nsat_mult', 'f_so']
        sigmas       = ['sigma_c', 'sigma_sA', 'sigma_sB', 'sigma_so']
        amplitudes   = ['NcBs', 'NsBsB', 'nbar', 'N']
        nuisance     = ['gamma_b1cB', 'gamma_b1sA', 'gamma_b1sB', 'delta_sigsA', 'delta_sigsB', 'f1h_sBsB', 'f1h_cBs']
        self.__dict__['valid_params'] = cosmo + biases + fractions + sigmas + amplitudes + nuisance
    
        # cosmology
        self.sigma8_z   = AttrDict(vary=False, fiducial=0.61, prior='uniform', lower=0.3, upper=0.9)
        self.f          = AttrDict(vary=False, fiducial=0.78, prior='uniform', lower=0.6, upper=1.0)
        self.alpha_perp = AttrDict(vary=False, fiducial=1.00, prior='uniform', lower=0.8, upper=1.2)
        self.alpha_par  = AttrDict(vary=False, fiducial=1.00, prior='uniform', lower=0.8, upper=1.2)
    
        # biases
        self.b1_cA      = AttrDict(vary=False, fiducial=1.90, prior='uniform', lower=1.2, upper=2.5)
        self.b1_cB      = AttrDict(vary=False, fiducial=2.84)
        self.b1_sA      = AttrDict(vary=False, fiducial=2.63)
        self.b1_sB      = AttrDict(vary=False, fiducial=3.62)
    
        # fractions
        self.fcB        = AttrDict(vary=False, fiducial=0.089, min=0, max=1)
        self.fsB        = AttrDict(vary=False, fiducial=0.400, prior='uniform', lower=0., upper=1., min=0., max=1)
        self.fs         = AttrDict(vary=False, fiducial=0.104, prior='uniform', lower=0., upper=0.25, min=0., max=1)
        self.Nsat_mult  = AttrDict(vary=False, fiducial=2.400, prior='normal', mu=2.4, sigma=0.2, min=2.)

        # sigmas
        self.sigma_c  = AttrDict(vary=False, fiducial=1., prior='uniform', lower=0., upper=3.)
        self.sigma_sA = AttrDict(vary=False, fiducial=3.5, prior='uniform', lower=2., upper=8.)
        self.sigma_sB = AttrDict(vary=False, fiducial=5, prior='uniform', lower=3., upper=10.)
    
        # amplitude
        self.NcBs  = AttrDict(vary=False, fiducial=4.5e4)
        self.NsBsB = AttrDict(vary=False, fiducial=9.45e4)
        self.nbar  = NBarParam(vary=False, fiducial=3.117e-4)
        self.N     = AttrDict(vary=False, fiducial=0, prior='uniform', lower=0, upper=500, min=0)

        # so vs fof
        self.f_so     = AttrDict(vary=False, fiducial=0.03, prior='normal', mu=0.04, sigma=0.02)
        self.sigma_so = AttrDict(vary=False, fiducial=4, prior='uniform', lower=1., upper=7)

        # nuisance
        self.gamma_b1cB = AttrDict(vary=False, fiducial=0.40, prior='normal', mu=0.4, sigma=0.2, min=0., max=1)
        self.gamma_b1sA = AttrDict(vary=False, fiducial=1.45, prior='normal', mu=1.45, sigma=0.3, min=1.0)
        self.gamma_b1sB = AttrDict(vary=False, fiducial=2.05, prior='normal', mu=2.05, sigma=0.3, min=1.0) 

        self.delta_sigsA = AttrDict(vary=False, fiducial=1., prior='normal', mu=1.0, sigma=0.2, min=0.)
        self.delta_sigsB = AttrDict(vary=False, fiducial=1., prior='normal', mu=1.0, sigma=0.2, min=0.)
        
        self.f1h_sBsB = AttrDict(vary=False, fiducial=4.0, prior='normal', mu=4.0, sigma=1.0, min=0.)
        self.f1h_cBs  = AttrDict(vary=False, fiducial=1.0, prior='normal', mu=1.0, sigma=0.75 , min=0)
        
        # model parameters
        self.__dict__['model'] = ModelParams()
        self.__dict__['_name'] = None
        self.__dict__['options'] = []
    
    def copy(self):
        return copy.copy(self)
        
    @property
    def name(self):
        """
        The name of the parameter set
        """
        if self._name is None:
            return self._name
        
        return "_".join([self._name] + sorted(self.options))
        
    @property
    def free_names(self):
        """
        Return a list of the free parameter names (alphabetically sorted)
        """
        return sorted(name for name in self.valid_params if getattr(self, name).vary)
        
    @property
    def fixed_names(self):
        """
        Return a list of the fixed parameter names that aren't constrained
        """
        toret = []
        for name in self.valid_params:
            par = getattr(self, name)
            if not par.vary and par.get('expr', None) is None:
                toret.append(name)
        return sorted(toret)
        
    @property
    def constrained_names(self):
        """
        Return a list of the constrained parameter names
        """
        toret = []
        for name in self.valid_params:
            par = getattr(self, name)
            if par.get('expr', None) is not None:
                toret.append(name)
        return sorted(toret)
    
    @property 
    def size(self):
        """
        Return the number of free parameters
        """
        return len(self.free_names)
    
    def to_file(self, ff):
        """
        Write out all valid parameters to the specified file object
        """
        # a few checks
        if self.model.z is None:
            raise ValueError("probably shouldn't write to file if `model.z` is `None`")
        if self.model.cosmo_filename is None:
            raise ValueError("probably shouldn't write to file if `model.cosmo_filename` is `None`")

        # free params
        ff.write("#{0}\n# free params\n#{0}\n".format("-"*78))
        for name in self.free_names:
            par = getattr(self, name)
            ff.write("theory.%s = %s\n" %(name, repr(par)))
        ff.write("\n#{0}\n# constrained params\n#{0}\n".format("-"*78))
        # constrained params
        for name in self.constrained_names:
            par = getattr(self, name)
            ff.write("theory.%s = %s\n" %(name, repr(par)))
        ff.write("\n#{0}\n# fixed params\n#{0}\n".format("-"*78))
        # fixed params
        for name in self.fixed_names:
            par = getattr(self, name)
            ff.write("theory.%s = %s\n" %(name, repr(par)))
        # model params
        ff.write("\n#{0}\n# model params\n#{0}\n".format("-"*78))
        ff.write(str(self.model))
            
    def __iter__(self):
        """
        Iterate over the free parameters
        """
        for name in self.free_names:
            yield name
            
    def __str__(self):
        """
        Print out a string representation
        """
        header = "%s\n%s\n\n" %(self.__class__.__name__, "-"*20)
        labels = ['Free parameters', 'Constrained parameters', 'Fixed parameters']
        atts = ['free_names', 'constrained_names', 'fixed_names']
        for tag, att in zip(labels, atts):
            x = getattr(self, att)
            if len(x):
                header += "%s\n%s\n" %(tag, '-'*20)
                params = []
                for name in getattr(self, att):
                    par = getattr(self, name)
                    values = ", ".join(["%s = %s" %(k, par[k]) for k in sorted(par)])
                    params.append("%-20s: %s" %(name, values))
                header += "\n".join(params)
                header += "\n\n"
        
        header += "%s\n%s\n" %('Model parameters', '-'*20)
        header += str(self.model)
        return header
    
    def __repr__(self):
        name = self.__class__.__name__
        return "<%s: %d free parameters>" %(name, self.size)
    
    def __getattribute__(self, key):
        try:
            return object.__getattribute__(self, key)
        except:
            args = (key, self.__class__.__name__)
            raise KeyError("`%s` is not a valid parameter name for class %s" %args)
        
    def __setattr__(self, key, value):
        if key not in self.valid_params:
            args = (key, self.__class__.__name__)
            raise KeyError("`%s` is not a valid parameter name for class %s" %args)
        
        # update the dict if the key exists
        if key in self.__dict__ and isinstance(value, dict):
            att = getattr(self, key)
            att.update(**value)
        # just set it
        else:
            object.__setattr__(self, key, value)
        
    def update(self, filename, ignore=[]):
        """
        Update the parameters from file with the synax `theory.param_name`
        """
        ns = {'theory':self, 'model': self.model}
        for name in ignore:
            ns[name] = AttrDict()
        if os.path.isfile(filename):
            with open(filename) as f:
                code = compile(f.read(), filename, 'exec')
            exec(code, ns)
        else:
            exec(filename, ns)
        
    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Return an instance of the class, adfter updating the parameters
        from the specified file
        """
        toret = cls()
        toret.update(filename, **kwargs)
        return toret
        
    def apply_options(self, *args):
        """
        Apply the various options
        """
        for arg in args:
            name = 'use_%s' %arg
            if name not in globals():
                raise ValueError("cannot find function `%s` to apply option" %name)
            globals()[name](self)
            
from .options import *
