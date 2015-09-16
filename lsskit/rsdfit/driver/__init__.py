from .. import AttrDict

class DriverParams(object):
    """
    Class to hold and manipulate the driver parameters required for fitting
    """
    valid_params = ['fitter', 'init_from', 'start_chain', 'burnin', 'test_convergence', 'epsilon']
    
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
        execfile(filename, ns)
        
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
    def fitter(self):
        """
        The type of fitter, default is `emcee`
        """
        try:
            return self._fitter
        except:
            return 'emcee'
    
    @fitter.setter
    def fitter(self, val):
        self._fitter = val
        
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
    def start_chain(self):
        """
        The name of the chain file to use to initialize the parameters
        """
        try:
            return self._start_chain
        except AttributeError:
            raise AttributeError("please specify `start_chain`")
    
    @start_chain.setter
    def start_chain(self, val):
        import os
        if not os.path.exists(val):
            raise RuntimeError("cannot set `start_chain` to `%s`: no such file" %val)
        if os.path.isdir(val):
            from glob import glob
            from pyRSD.rsdfit.results import EmceeResults
            import operator
            
            pattern = os.path.join(val, "*.npz")
            chains = glob(pattern)
            if not len(chains):
                raise RuntimeError("did not find any chain (`.npz`) files matching pattern `%s`" %pattern)
            
            # find the chain file which has the maximum log prob in it and use that
            max_lnprobs = [EmceeResults.from_npz(f).max_lnprob for f in chains]
            index, value = max(enumerate(max_lnprobs), key=operator.itemgetter(1))
            self._start_chain = chains[index]
        
    
        
    
            
    
    
            
    
    
        