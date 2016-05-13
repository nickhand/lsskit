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

from . import DriverParams, BaseTheoryParams
from . import PkmuDataParams, PoleDataParams
from . import RSDFIT_BIN, RSDFIT_PARAMS

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
    
def construct_rsdfit_call(config, stat, kmax, 
                            theory_options=[], rsdfit_options=[], tag="", command=None):
    """
    Construct and return the string representation of the ``rsdfit`` command
    
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
    # write out the parameters to a temporary file
    ff = tempfile.NamedTemporaryFile(delete=False)
    param_file = ff.name
    
    # initialize the driver, theory, and data parameter objects
    driver = DriverParams.from_file(config, ignore=['theory', 'model', 'data'])
    theory = BaseTheoryParams.from_file(config, ignore=['driver', 'data'])
    if stat == 'pkmu':
        data = PkmuDataParams.from_file(config, ignore=['theory', 'model', 'driver'])
    else:
        data = PoleDataParams.from_file(config, ignore=['theory', 'model', 'driver'])
            
    # apply any options to the theory model
    if theory_options is not None:
        theory.apply_options(*theory_options)
        
    # set the kmax
    data.kmax = kmax

    # write out the parameters to the specified output
    driver.to_file(ff)
    data.to_file(ff)
    theory.to_file(ff)
    ff.close()
    
    # get the output name
    tags = [x.name for x in [data, theory, driver] if x.name]
    if tag: tags.append(tag)
    output_dir = os.path.join(driver.output, driver.solver_type + '_' + "_".join(tags))
    
    # add the model?
    if driver.model_file is not None:
        if all(x not in rsdfit_options for x in ['-m', '--model']):
            rsdfit_options += ['-m', driver.model_file]

    # add extra param file?
    if all(x not in rsdfit_options for x in ['-xp', '--extra_params']):
        xparams = os.path.join(RSDFIT_PARAMS, 'general', 'extra.params')
        if os.path.isfile(xparams):
            rsdfit_options += ['-xp', xparams]

    # the executable command
    if command is None:
        command = RSDFIT_BIN
        
    # return the call signature
    return command.split() + ['run', '-o', output_dir, '-p', param_file] + rsdfit_options
   
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
    kws = {'theory_options':theory_options, 'rsdfit_options':rsdfit_options, 
            'tag':tag, 'command':command}
    call_signature = construct_rsdfit_call(config, stat, kmax, **kws)
    
    # run the command
    if nodes is None and partition is None:
        try:
            ret = subprocess.call(call_signature)
            if ret: raise
        except Exception as e:
            if not isinstance(e, KeyboardInterrupt):        
                msg = "error calling ``rsdfit``:\n" + "-"*20 + "\n"
                msg += "\toriginal command:\n\t\t%s\n" %(" ".join(call_signature))
                msg += "\toriginal exception: %s\n" %e.msg
                e.msg = msg
            raise e
        finally:
            i = call_signature.index('-p')
            param_file = call_signature[i+1]
            if os.path.exists(param_file):
               os.remove(param_file)
    # submit the job
    else:
        if nodes is None or partition is None:
            raise ValueError("both `nodes` and `partition` must be given to submit job")
        
        command = " ".join(call_signature)
        submit_rsdfit_job(command, nodes, partition)

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