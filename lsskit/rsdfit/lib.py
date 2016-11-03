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
from .command import RSDFitCommand

def slurm_time(s):
    """
    Format the input minutes to hours:minutes:seconds
    """
    hours, minutes = divmod(int(s), 60)
    return "%02d:%02d:00" %(hours, minutes)

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
                theory_options=[], rsdfit_options=[], tag="", start=None,
                command=None, nodes=None, partition=None, time=None, print_output=False):
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
    theory_options : list, optional
        list of options to apply the theory model, i.e., `mu_corr` or `so_corr`
    rsdfit_options : list, optional
        list of additional options to pass to the ``rsdfit`` command
    tag : str, optional
        the name of the tag to append to the output directory
    start : str,optional
        set ``start_from`` to this value, if not `None`
    command : str, optional
        the executable command to call
    nodes : int, optional
        the number of nodes to ask for when submitting the job
    partition : str, optional
        the partition to submit the job to
    time : str, optional
        the amount of time to request for the job
    print_output : bool, optional
        just print the intended output directory and exit
    """
    # make the call signature
    kws = {}
    kws['theory_options'] = theory_options
    kws['options'] = rsdfit_options 
    kws['tag'] = tag
    kws['executable'] = command
    kws['start_from'] = start
    
    # create the command
    with RSDFitCommand(config, stat, kmax, **kws) as command:
    
        # just print the output and return
        if print_output:
            print(command.output_dir)
            if os.path.exists(command.param_file):
               os.remove(command.param_file)
            return
        
        # run the command
        if nodes is None and partition is None and time is None:
            try:
                p = subprocess.Popen(command(), stdout=subprocess.PIPE)
                output = p.communicate()[0]
                if p.returncode: raise
            except Exception as e:
                if e is None: e = Exception ()
                if not isinstance(e, KeyboardInterrupt):        
                    msg = "error calling ``rsdfit``:\n" + "-"*20 + "\n"
                    msg += "\toriginal command:\n\t\t%s\n" %(command)
                    msg += "\toriginal exception: %s\n" %str(e)
                    e.msg = msg
                raise e
            finally:
                if os.path.exists(command.param_file):
                   os.remove(command.param_file)
        
        # submit the job
        else:
            if nodes is None or partition is None or time is None:
                raise ValueError("`nodes`, `partition`, and `time` must all be given to submit job")

            submit_rsdfit_job(str(command), nodes, partition, time)

def submit_rsdfit_job(command, nodes, partition, time):
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
    time : str
        the amount of time to request in format `hours:minutes:seconds`
    """
    batch_file = """#!/bin/bash
    
    module unload python
    module load python/3.5-anaconda
    source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh
    bcast -v $TAR_DIR/$NERSC_HOST/pyRSD*
    
    N=$(($CPUS_PER_NODE * $SLURM_NNODES))
    %s
    """
    # get the slurm output dir
    output_dir = os.path.join(os.path.abspath(os.path.curdir), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # first figure out the NERSC host
    host = subprocess.check_output(["printenv", "NERSC_HOST"]).strip()
    host = host.decode('utf-8')
    o = output_dir + os.sep + "slurm-{0}-%j.out".format(host)
    
    if host == 'edison':
        command = "srun -n $N %s" %command
    elif host == 'cori':
        command = "srun -n $N -c 1 %s" %command
    else:
        raise ValueError("host '%s' not understood" %host)
    
    batch_file = batch_file %command
    sbatch_cmd = ['sbatch', '-N', str(nodes), '-p', partition, '-t', time, '-o', o]
    if host == 'cori':
        sbatch_cmd += ['-C', 'haswell']
    
    p = subprocess.Popen(sbatch_cmd, stdin=subprocess.PIPE)
    p.communicate(batch_file.encode())
    
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
    
def find_batch_parameters(config, keys):
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
    lines = config.split("\n")
    for line in lines:
        
        li = line.strip()
        if not li.startswith("#"):
            fields = li.split('=')
            if len(fields) == 2:
                
                names = fields[0].strip().split('.')
                subkey = '.'.join(names[1:])
                value = fields[1].strip()
                
                # check if this is valid key
                if any("{%s}" %k in value for k in keys):
                    value = eval(value)
                    tup = BatchParam(key=names[0], subkey=subkey, value=value)
                    toret.append(tup)
                
    return toret
