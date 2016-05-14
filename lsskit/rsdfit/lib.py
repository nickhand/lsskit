"""
    run_rsdfit.py
    lsskit.rsdfit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : run rsdfit with the parameters specified on the command line
"""

import os, sys
import tempfile
import subprocess
import string
from collections import namedtuple
import copy

from . import DriverParams, BaseTheoryParams
from . import PkmuDataParams, PoleDataParams
from . import RSDFIT_BIN, RSDFIT_PARAMS

class RSDFitCommand(object):
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
        # initialize the driver, theory, and data parameter objects
        self.driver = DriverParams.from_file(config, ignore=['theory', 'model', 'data'])
        self.theory = BaseTheoryParams.from_file(config, ignore=['driver', 'data'])
        if stat == 'pkmu':
            self.data = PkmuDataParams.from_file(config, ignore=['theory', 'model', 'driver'])
        else:
            self.data = PoleDataParams.from_file(config, ignore=['theory', 'model', 'driver'])
            
        # apply any options to the theory model
        if theory_options is not None:
            self.theory.apply_options(*theory_options)
        
        # set the kmax
        self.data.kmax = kmax
        
        # store the rest
        self.tag = tag
        self.executable = executable
        self.input_options = options
        if self.executable is None: self.executable = RSDFIT_BIN
        
        # initialize the components
        self.output_dir = None
        self.param_file = None
        self.args       = None
    
    def copy(self):
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
    
        # copy the options
        self.args = list(self.input_options)
        
        # now add the rest of the options
        self.args += ['run', '-o', self.output_dir, '-p', self.param_file]
        
        # add the model?
        if self.driver.model_file is not None:
            if all(x not in self.args for x in ['-m', '--model']):
                self.args += ['-m', self.driver.model_file]

        # add extra param file?
        if all(x not in self.args for x in ['-xp', '--extra_params']):
            xparams = os.path.join(RSDFIT_PARAMS, 'general', 'extra.params')
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


def write_rsdfit_params(mode, config, output, theory_options=[]):
    """
    Write a parameter file for ``rsdfit`` from a configuration file
        
    Parameters
    ----------
    mode : str, {`pkmu`, `poles`}
        either `pkmu` or `poles` -- the mode of the RSD fit
    config : str
        the name of the file holding the configuration parameters, 
        from which the the parameter file for ``rsdfit`` will be made
    output : file object
        the file object to write to
    theory_options : list
        list of options to apply the theory model, i.e., `mu_corr` or `so_corr`
    """
    # initialize the driver, theory, and data parameter objects
    driver = DriverParams.from_file(config, ignore=['theory', 'model', 'data'])
    theory = BaseTheoryParams.from_file(config, ignore=['driver', 'data'])
    if mode == 'pkmu':
        data = PkmuDataParams.from_file(config, ignore=['theory', 'model', 'driver'])
    else:
        data = PoleDataParams.from_file(config, ignore=['theory', 'model', 'driver'])

    # apply any options to the theory model
    if theory_options is not None:
        theory.apply_options(*theory_options)

    # write out the parameters to the specified output
    driver.to_file(output)
    data.to_file(output)
    theory.to_file(output)
       
def run_rsdfit(config, stat, kmax, 
                theory_options=[], rsdfit_options=[], tag="", 
                command=None, nodes=None, partition=None):
    """
    Run ``rsdfit``, or the return the call signature. This constructs the ``rsdfit``
    parameter file the from input arguments.
    
    Notes
    -----
    *   if the configuration file specifies a model file, this will be appended
        to the rsdfit call with the `-m` option
    *   the parameter file is written to a temporary file, which is not deleted
        by the code -- perhaps we can fix this?
    
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
    rsdfit_options : list
        list of additional options to pass to the ``rsdfit`` command
    tag : str
        the name of the tag to append to the output directory
    command : str
        the executable command to call
    """
    # make the call signature
    kws = {}
    kws['theory_options'] = theory_options
    kws['options'] = rsdfit_options, 
    kws['tag'] = tag
    kws['executable'] = command
    
    # create the command
    with RSDFitCommand(config, stat, kmax, **kws) as command:
    
        # run the command
        if nodes is None and partition is None:
            try:
                ret = subprocess.call(command())
                if ret: raise
            except Exception as e:
                if not isinstance(e, KeyboardInterrupt):        
                    msg = "error calling ``rsdfit``:\n" + "-"*20 + "\n"
                    msg += "\toriginal command:\n\t\t%s\n" %(command)
                    msg += "\toriginal exception: %s\n" %e.msg
                    e.msg = msg
                raise e
            finally:
                if os.path.exists(command.param_file):
                   os.remove(command.param_file)
        
        # submit the job
        else:
            if nodes is None or partition is None:
                raise ValueError("both `nodes` and `partition` must be given to submit job")

            submit_rsdfit_job(str(command), nodes, partition)

def submit_rsdfit_job(command, nodes, partition):
    """
    Submit a ``rsdfit`` job via the SLURM batch scheduling system
    
    Parameters
    ----------
    command : str
        the ``rsdfit`` command to run
    nodes : int
        the number of nodes to request
    partition : str
        the queue to submit the job to
    """
    batch_file = """#!/bin/bash
    
    source /project/projectdirs/m779/python-mpi/nersc/activate.sh
    bcast -v $TAR_DIR/$NERSC_HOST/pyRSD*
    
    N=$(($CPUS_PER_NODE * $SLURM_NNODES))
    srun -n $N %s
    """
    batch_file = batch_file %command
    sbatch_cmd = ['sbatch', '-N', str(nodes), '-p', partition]
    
    p = subprocess.Popen(sbatch_cmd, stdin=subprocess.PIPE)
    p.communicate(batch_file)
    
def MyStringParse(formatter, s, keys):
    """
    Special string format parser that will only 
    format certain keys, doing nothing with
    others
    
    Parameters
    ----------
    formatter : string.Formatter
        the base formatter class
    s : str
        the string we are formatting
    keys: list
        the list of keys we are formatting, ignoring
        all other keys
    """
    l = list(string.Formatter.parse(formatter, s))
    toret = []
    for x in l:
        if x[1] in keys:
            toret.append(x)
        else:
            val = x[0]
            if x[1] is not None:
                fmt = "" if not x[2] else ":%s" %x[2]
                val += "{%s%s}" %(x[1], fmt)
            toret.append((val, None, None, None))
    return iter(toret)
    
def make_temp_config(config, key, value):
    """
    Make a temporary configuration file, by string formatting
    the input string and writing to a temporary file
    
    Parameters
    ----------
    config : str
        the lines of the input template configuration file, 
        returned via read()
    key : list
        list of the names for each iteration dimension that
        serve as the string formatting keys
    values : list
        list of tuples providing the string formatting values
        for each iteration
    
    Returns
    -------
    str :
        the name of the temporary file that was written to
    """
    # initialize a special string formatter
    formatter = string.Formatter()
    formatter.parse = lambda l: MyStringParse(formatter, l, key)
    
    d = dict(zip(key, value))
    with tempfile.NamedTemporaryFile(delete=False) as ff:
        fname = ff.name
        ff.write(formatter.format(config, **d))
        
    return fname
    
def read_batch_parameters(fname):
    """
    Read the parameters that will be updated for each iteration in 
    batch mode, from a template configuration file
    
    Parameters
    ----------
    fname : str
        the name of the file to read from

    Returns
    -------
    list of BatchParam :
        a list of named tuples that have `key`, `subkeys`, and `value`
        fields
    """
    BatchParam = namedtuple('BatchParam', ['key', 'subkey', 'value'])
    
    toret = []
    lines = open(fname, 'r').readlines()
    for line in lines:
        
        li = line.strip()
        if not li.startswith("#"):
            fields = li.split('=')
            if len(fields) == 2:
                
                names = fields[0].strip().split('.')
                subkey = '.'.join(names[1:])
                value = eval(fields[1].strip())
                
                tup = BatchParam(key=names[0], subkey=subkey, value=value)
                toret.append(tup)
                
    return toret
