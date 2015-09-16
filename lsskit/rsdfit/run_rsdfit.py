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

    h = 'additional theory parameter options to impose'
    parser.add_argument('--options', type=str, nargs='*', help=h)

    h = 'the name of the python command to run; default is just `rsdfit`'
    parser.add_argument('--command', type=str, default='rsdfit', help=h)

    ns, other = parser.parse_known_args()
    run_rsdfit(run=True, rsdfit_options=other, **vars(ns))

def run_rsdfit(mode=None, config=None, options=[], command=None, run=True, rsdfit_options=[]):
    """
    Run rsdfit with the parameters specified on the command line
    """
    
    # initialize the parameter objects
    driver = DriverParams.from_file(config, ignore=['theory', 'model', 'data'])
    theory = BaseTheoryParams.from_file(config, ignore=['driver', 'data'])
    if mode == 'pkmu':
        data = PkmuDataParams.from_file(config, ignore=['theory', 'model', 'driver'])
    else:
        data = PoleDataParams.from_file(config, ignore=['theory', 'model', 'driver'])

    # apply any options to the theory model
    if options is not None:
        theory.apply_options(*options)

    # write the temporary param file
    with tempfile.NamedTemporaryFile(delete=False) as ff:
        param_file = ff.name
        driver.to_file(ff)
        data.to_file(ff)
        theory.to_file(ff)
    
    # get the output name
    tags = [x.name for x in [driver, theory, data] if x.name]
    output_dir = os.path.join(driver.output, "_".join(tags))

    # the options to pass to rsdfit
    call_signature = command.split() + ['run', '-o', output_dir, '-p', param_file] + rsdfit_options
    if run:
        try:
            ret = subprocess.call(call_signature)
            if ret:
                raise RuntimeError
        except:
            raise RuntimeError("error calling command: %s" %" ".join(call_signature))
        finally:
            if os.path.exists(param_file):
                os.remove(param_file)
    else:
        print call_signature
        return " ".join(map(str, call_signature))
            
            
if __name__ == '__main__':
    main()



    




