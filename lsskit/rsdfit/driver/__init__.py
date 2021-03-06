from .. import AttrDict, os
import copy


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

class DriverParams(object):
    """
    Class to hold and manipulate the driver parameters required for fitting
    """
    valid_params = ['solver_type', 'init_from', 'start_from', 'burnin', 'test_convergence', 'epsilon',
                     'lbfgs_epsilon', 'lbfgs_use_priors', 'lbfgs_options', 'init_scatter']
    
    def copy(self):
        return copy.copy(self)
        
    def __iter__(self):
        """
        Iterate over the valid parameters
        """
        for name in self.valid_params:
            yield name
    
    def __str__(self):
        toret = []
        for k in self:
            v = getattr(self, k, None)
            toret.append("driver.%s = %s" %(k,repr(v)))
        return "\n".join(toret)
        
    def update(self, filename, ignore=[]):
        """
        Update the parameters from file with the synax `data.param_name`
        """
        ns = {'driver':self}
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
        
    def to_file(self, ff):
        """
        Write out all valid parameters to the specified file object
        """
        ff.write("#{0}\n# driver params\n#{0}\n".format("-"*78))
        for name in self:
            if not hasattr(self, name):
                raise ValueError('please specify `%s` attribute before writing to file' %name)
            par = getattr(self, name)
            ff.write("driver.%s = %s\n" %(name, repr(par)))
                        
    @property
    def solver_type(self):
        """
        The type of solver_type, default is `mcmc`
        """
        try:
            return self._solver_type
        except:
            return 'mcmc'
    
    @solver_type.setter
    def solver_type(self, val):
        self._solver_type = val
        
    @property
    def output(self):
        """
        The output directory will the run directory will live
        """
        try:
            return self._output
        except:
            return '.'
    
    @output.setter
    def output(self, val):
        if '$' in val:
            val = replace_vars(val, {})
        self._output = val
        
    @property
    def model_file(self):
        """
        The name of the file holding the model
        """
        try:
            return self._model_file
        except:
            raise None
    
    @model_file.setter
    def model_file(self, val):
        
        if '$' in val:
            val = replace_vars(val, {})    
        self._model_file = val
        
    @property
    def burnin(self):
        """
        The number of steps to consider ``burnin``; defaults to 0
        """
        try:
            return self._burnin
        except:
            return 0
    
    @burnin.setter
    def burnin(self, val):
        self._burnin = val
        
    @property
    def test_convergence(self):
        """
        Whether to test the convergence between multiple chains when running; default is False
        """
        try:
            return self._test_convergence
        except:
            return False
    
    @test_convergence.setter
    def test_convergence(self, val):
        self._test_convergence = val
        
    @property
    def epsilon(self):
        """
        The degree of accuracy required for convergence; default is 0.02
        """
        try:
            return self._epsilon
        except:
            return 0.02
    
    @epsilon.setter
    def epsilon(self, val):
        self._epsilon = val
        
    @property
    def lbfgs_epsilon(self):
        """
        The step-size for derivatives in LBFGS; default is 1e-4
        """
        try:
            return self._lbfgs_epsilon
        except:
            return 1e-4
    
    @lbfgs_epsilon.setter
    def lbfgs_epsilon(self, val):
        self._lbfgs_epsilon = val
        
    @property
    def init_scatter(self):
        """
        The fractional (normal) scatter to add to the init values
        """
        try:
            return self._init_scatter
        except:
            return 0.
    
    @init_scatter.setter
    def init_scatter(self, val):
        self._init_scatter = val
        
    @property
    def lbfgs_options(self):
        """
        The options in LBFGS
        """
        try:
            return self._lbfgs_options
        except:
            return {}
    
    @lbfgs_options.setter
    def lbfgs_options(self, val):
        self._lbfgs_options = val
        
    @property
    def lbfgs_use_bounds(self):
        """
        Whether to use bounds in LBFGS; default is False
        """
        try:
            return self._lbfgs_use_bounds
        except:
            return False
    
    @lbfgs_use_bounds.setter
    def lbfgs_use_bounds(self, val):
        self._lbfgs_use_bounds = val
        
    @property
    def lbfgs_use_priors(self):
        """
        Whether to use priors in LBFGS; default is True
        """
        try:
            return self._lbfgs_use_priors
        except:
            return True
    
    @lbfgs_use_priors.setter
    def lbfgs_use_priors(self, val):
        self._lbfgs_use_priors = val
        
    @property
    def name(self):
        """
        The identifying name for this parameter set
        """
        try:
            return self._name
        except:
            return ''
    
    @name.setter
    def name(self, val):
        self._name = val
        
    @property
    def start_from(self):
        """
        The name of the results file to use to initialize the parameters
        """
        try:
            return self._start_from
        except AttributeError:
            raise AttributeError("please specify `start_from`")
    
    @start_from.setter
    def start_from(self, val):
        if val is None:
            self._start_from = val
            return
            
        if '$' in val:
            val = replace_vars(val, {})
        
        import os
        if not os.path.exists(val):
            raise RuntimeError("cannot set `start_from` to `%s`: no such file" %val)
        
        if os.path.isdir(val):
            from glob import glob
            from pyRSD.rsdfit.results import EmceeResults, LBFGSResults
            import operator
            
            pattern = os.path.join(val, "*.npz")
            result_files = glob(pattern)
            if not len(result_files):
                raise RuntimeError("did not find any chain (`.npz`) files matching pattern `%s`" %pattern)
            
            # find the chain file which has the maximum log prob in it and use that
            max_lnprobs = []
            for f in result_files:
                
                try:
                    r = EmceeResults.from_npz(f)
                    max_lnprobs.append(r.max_lnprob)
                except:
                    r = LBFGSResults.from_npz(f)
                    max_lnprobs.append(-r.min_chi2)

            index, value = max(enumerate(max_lnprobs), key=operator.itemgetter(1))
            self._start_from = result_files[index]
        else:
            self._start_from = val
        
    
        
    
            
    
    
            
    
    
        