"""
    run_rsdfit.py
    lsskit.rsdfit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : run rsdfit with the parameters specified on the command line
"""
import argparse
import os
import tempfile
import subprocess
from lsskit.rsdfit import DriverParams, BaseTheoryParams, \
                            PkmuDataParams, PoleDataParams
                            
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

    # write out the parameters to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as ff:
        param_file = ff.name
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
            
            
def main():
    """
    Run as a console script
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



    




