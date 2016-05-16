import tempfile
import string
import os

from . import RSDFIT_BIN, RSDFIT_BATCH, RSDFIT
from . import DriverParams, BaseTheoryParams
from . import PkmuDataParams, PoleDataParams

class BaseCommand(object):
    """
    Base class to represent a ``rsdfit`` command
    """
    def __init__(self, config, stat, kmax, 
                    theory_options=[], options=[], tag="", executable=None):
    
        # make sure the config file exists
        if not os.path.isfile(config):
            raise ValueError("the input configuration file does not exist")
    
        # just store the options
        self.config         = config
        self.stat           = stat
        self.kmax           = kmax
        self.theory_options = theory_options
        self.input_options  = options
        self.tag            = tag
        self.executable     = executable            
                            
        # initialize the components
        self.output_dir = None
        self.param_file = None
        self.args       = None
    
    def copy(self):
        """
        Return a copy
        """
        return copy.copy(self)    
    
    @property
    def initialized(self):
        """
        Whether the commmand has been properly initialized
        """
        return self.param_file is not None
    
    def __enter__(self):
        """
        Initialize the temporary file and write the parameters
        """
        # write out the parameters to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as ff:
            self.param_file = ff.name
        
            # write out the parameters to the specified output
            self.driver.to_file(ff)
            self.data.to_file(ff)
            self.theory.to_file(ff)
            
        # get the output name
        tags = [x.name for x in [self.data, self.theory, self.driver] if x.name]
        if self.tag: tags.append(self.tag)
        self.output_dir = os.path.join(self.driver.output, self.driver.solver_type + '_' + "_".join(tags))
            
        # initialize the options
        self.args = ['run', '-o', self.output_dir, '-p', self.param_file]
        
        # add the input options specified
        self.args += list(self.input_options)
        
        # add the model?
        if self.driver.model_file is not None:
            if all(x not in self.args for x in ['-m', '--model']):
                self.args += ['-m', self.driver.model_file]

        # add extra param file?
        if all(x not in self.args for x in ['-xp', '--extra_params']):
            xparams = os.path.join(RSDFIT, 'params', 'general', 'extra.params')
            if os.path.isfile(xparams):
                self.args += ['-xp', xparams]

        # return
        return self
        
    def __exit__(self, *args):
        """
        Reset the command
        """
        # reset
        self.output_dir = None
        self.param_file = None
        self.args       = None

    def __str__(self):
        """
        The string representation of the command
        """
        return " ".join(self.__call__())
    
    def __call__(self):
        """
        Return the `split` command as a list
        """
        if not self.initialized:
            raise ValueError("``rsdfit`` command not currently initialized")
        
        return self.executable.split() + self.args
        

class RSDFitCommand(BaseCommand):
    """
    Class to represent a ``rsdfit`` command
    """
    def __init__(self, config, stat, kmax, 
                    theory_options=[], options=[], tag="", executable=None):
        """    
        Parameters
        ----------
        config : str
            the name of the file holding the configuration parameters, from which the
            the parameter file for ``rsdfit`` will be initialized
        stat : str, {`pkmu`, `poles`}
            either `pkmu` or `poles` -- the mode of the RSD fit
        kmax : list
            the kmax values to use
        theory_options : list
            list of options to apply the theory model, i.e., `mu_corr` or `so_corr`
        options : list
            list of additional options to pass to the ``rsdfit`` command
        tag : str
            the name of the tag to append to the output directory
        executable : str
            the executable command to call

        """    
        # initial the base class
        kws = {'theory_options':theory_options, 'options':options, 'tag':tag, 'executable':executable}
        super(RSDFitCommand, self).__init__(config, stat, kmax, **kws)
                    
        # initialize the driver, theory, and data parameter objects
        self.driver = DriverParams.from_file(self.config, ignore=['theory', 'model', 'data'])
        self.theory = BaseTheoryParams.from_file(self.config, ignore=['driver', 'data'])
        if self.stat == 'pkmu':
            self.data = PkmuDataParams.from_file(self.config, ignore=['theory', 'model', 'driver'])
        else:
            self.data = PoleDataParams.from_file(self.config, ignore=['theory', 'model', 'driver'])
            
        # apply any options to the theory model
        if self.theory_options is not None and len(self.theory_options):
            self.theory.apply_options(*self.theory_options)
        
        # set the kmax
        self.data.kmax = self.kmax
        
        # update the executable
        if self.executable is None: 
            self.executable = RSDFIT_BIN
        
class RSDFitBatchCommand(BaseCommand):
    """
    Class to represent a ``rsdfit`` command in batch mode
    """
    def __init__(self, config, stat, kmax, 
                    theory_options=[], options=[], tag="", executable=None):
        """    
        Parameters
        ----------
        config : str
            the name of the template file holding the configuration parameters, which
            will be updated during each iteration
        stat : str, {`pkmu`, `poles`}
            either `pkmu` or `poles` -- the mode of the RSD fit
        kmax : list
            the kmax values to use
        theory_options : list
            list of options to apply the theory model, i.e., `mu_corr` or `so_corr`
        options : list
            list of additional options to pass to the ``rsdfit`` command
        tag : str
            the name of the tag to append to the output directory
        executable : str
            the executable command to call
        """    
        # initial the base class
        kws = {'theory_options':theory_options, 'options':options, 'tag':tag, 'executable':executable}
        super(RSDFitBatchCommand, self).__init__(config, stat, kmax, **kws)
        
        # read the template file into a string and store
        self.config = open(self.config, 'r').read()
        
        # update the executable
        if self.executable is None: 
            self.executable = RSDFIT_BATCH
            
    def update(self, kwargs, formatter=None):
        """
        Update the template config file.
        
        This is designed to be used as:
        
        >> with command.update(kwargs, formatter) as c:
        >>  .... (do stuff with ``c``) ....
         
        Parameters
        ----------
        kwargs : dict
            the string formatting keyword dictionary which will be applied
            to the template config file
        formatter : string.Formatter, optional
            a Formatter class that will be used to do the string formatting;
            if `None`, the usual `string.format` is used
        """
        # format the config first
        try:
            if formatter is not None:
                config = formatter.format(self.config, **kwargs)
            else:
                config = self.config.format(**kwargs)
        except Exeception as e:
            raise RuntimeError("error trying to format batch template config file: %s" %str(e))
                    
        # initialize the driver, theory, and data parameter objects
        self.driver = DriverParams.from_file(config, ignore=['theory', 'model', 'data'])
        self.theory = BaseTheoryParams.from_file(config, ignore=['driver', 'data'])
        if self.stat == 'pkmu':
            self.data = PkmuDataParams.from_file(config, ignore=['theory', 'model', 'driver'])
        else:
            self.data = PoleDataParams.from_file(config, ignore=['theory', 'model', 'driver'])
            
        # apply any options to the theory model
        if self.theory_options is not None and len(self.theory_options):
            self.theory.apply_options(*self.theory_options)
        
        # set the kmax
        self.data.kmax = self.kmax
        
        return self