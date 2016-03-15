"""
    run_rsdfit.py
    lsskit.rsdfit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : run rsdfit with the parameters specified on the command line
"""
import argparse
import os, sys
import tempfile
import subprocess
from lsskit.rsdfit import (DriverParams, BaseTheoryParams, 
                            PkmuDataParams, PoleDataParams)

from lsskit.rsdfit.theory import valid_theory_options

def ExistingFile(f):
    if not os.path.isfile(f):
        raise ValueError("'%s' is not an existing file" %f)
    return f
    
def _write_rsdfit_params():
    """
    The function to write an ``rsdfit`` parameter file, as called
    by a console script
    """
    desc = "write a parameter file for ``rsdfit`` from the input configuration file"
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('mode', choices=['pkmu', 'poles'], 
        help='the fitting mode; either ``pkmu`` or ``poles``')
    parser.add_argument('config', type=ExistingFile, 
        help='the configuration file to make the parameter file from')
    parser.add_argument('-th', '--theory_options', nargs='*', choices=valid_theory_options, 
        help='additional theory options to apply, i.e., `so_corr`')
    parser.add_argument('-o', '--output', type=argparse.FileType('w'), nargs='?',
        default=sys.stdout, help='the output file to write; default is stdout')
        
    ns = parser.parse_args()
    write_rsdfit_params(ns.mode, ns.config, ns.output, theory_options=ns.theory_options)
    

def write_rsdfit_params(mode, config, output, theory_options=[]):
    """
    Write a parameter file for ``rsdfit`` from a configuration file
        
    Parameters
    ----------
    mode : str, {`pkmu`, `poles`}
        either `pkmu` or `poles` -- the mode of the RSD fit
    config : str
        the name of the file holding the configuration parameters, from which the
        the parameter file for ``rsdfit`` will be made
    output : file object
        the file object to write to
    theory_options : list
        list of options to apply the theory model, i.e., `mu6_corr` or `so_corr`
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
    
    
def run_rsdfit(mode=None, config=None, theory_options=[], command=None, run=True, rsdfit_options=[]):
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
    mode : str, {`pkmu`, `poles`}
        either `pkmu` or `poles` -- the mode of the RSD fit
    config : str
        the name of the file holding the configuration parameters, from which the
        the parameter file for ``rsdfit`` will be initialized
    theory_options : list
        list of options to apply the theory model, i.e., `mu6_corr` or `so_corr`
    command : str
        the ``rsdfit`` command to run, i.e., ``python rsdfit.py`` or ``rsdfit``
    run : bool
        if `True`, run the ``rsdfit`` command, otherwise, just return
        the command to be run from a job script
    rsdfit_options : list
        list of additional options to pass to the ``rsdfit`` command
        
    Returns
    -------
    command : str
        if `run == False`, this returns the rsdfit command as a string
    """
    # write out the parameters to a temporary file
    ff = tempfile.NamedTemporaryFile(delete=False)
    param_file = ff.name
    
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
    driver.to_file(ff)
    data.to_file(ff)
    theory.to_file(ff)
    
    # get the output name
    tags = [x.name for x in [driver, theory, data] if x.name]
    output_dir = os.path.join(driver.output, "_".join(tags))
    
    # add the model?
    if driver.model_file is not None:
        if all(x not in rsdfit_options for x in ['-m', '--model']):
            rsdfit_options += ['-m', driver.model_file]

    # the options to pass to rsdfit
    call_signature = command.split() + ['run', '-o', output_dir, '-p', param_file] + rsdfit_options
    if run:
        try:
            ret = subprocess.call(call_signature)
            if ret: raise RuntimeError
        except:
            raise RuntimeError("error calling command: %s" %" ".join(call_signature))
        finally:
            if os.path.exists(param_file):
                os.remove(param_file)
    else:
        
        return " ".join(map(str, call_signature))
            
            
def _run_rsdfit():
    """
    Run ``run_rsdfit`` as a console script
    """
    desc = "run rsdfit with the parameters specified on the command line"
    parser = argparse.ArgumentParser(description=desc)

    h = 'the mode; either pkmu or poles'
    parser.add_argument('mode', choices=['pkmu', 'poles'], help=h)

    h = 'the name of the file holding the configuration parameters for this run'
    parser.add_argument('config', type=str, help=h)

    h = 'additional options to apply the theory model, i.e., `mu6_corr` or `so_corr`'
    parser.add_argument('--theory_options', type=str, nargs='*', help=h)

    h = 'the name of the python command to run; default is just `rsdfit`'
    parser.add_argument('--command', type=str, default='rsdfit', help=h)

    ns, other = parser.parse_known_args()
    run_rsdfit(run=True, rsdfit_options=other, **vars(ns))

if __name__ == '__main__':
    main()



    




