"""
    __main__.py
    lsskit.rsdfit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : module to hold the top-level console scripts for rsdfit
"""
import argparse 
import os

from lsskit.rsdfit import lib
from lsskit.rsdfit.theory import valid_theory_options

def ExistingFile(f):
    """
    An argparse argument type for an existing file
    """
    if not os.path.isfile(f):
        raise ValueError("'%s' is not an existing file" %f)
    return f

def write_rsdfit_params():
    """
    Write a parameter file for ``rsdfit`` from the input 
    configuration file
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
    lib.write_rsdfit_params(ns.mode, ns.config, ns.output, theory_options=ns.theory_options)

def run_rsdfit():
    """
    Run the ``rsdfit`` algorithm
    """
    desc = "run `rsdfit` with the parameters specified on the command line"
    parser = argparse.ArgumentParser(description=desc)
    
    h = 'the name of the template configuration file'
    parser.add_argument('config', type=str, help=h)
    
    h = 'the executable command to call, i.e, ``rsdfit`` or ``mpirun -n 2 rsdfit``'
    parser.add_argument('--command', type=str, help=h)
    
    # required named arguments
    group = parser.add_argument_group('configuration arguments')
    
    h = 'the statistic; either pkmu or poles'
    group.add_argument('--stat', choices=['pkmu', 'poles'], required=True, help=h)
    
    h = 'the maximum k to use'
    group.add_argument('--kmax', type=float, nargs='*', required=True, help=h)
    
    h = 'additional options to apply the theory model, i.e., `mu_corr` or `so_corr`'
    group.add_argument('-th', '--theory_options', type=str, nargs='*', help=h)
    
    h = 'additional tag to append to the output directory'
    group.add_argument('--tag', type=str, default="", help=h)
    
    # parse known and unknown arguments
    ns, other = parser.parse_known_args()
    
    kws = {'rsdfit_options':other, 'theory_options':ns.theory_options, 
            'tag':ns.tag, 'command':ns.command}
    lib.run_rsdfit(ns.config, ns.stat, ns.kmax, **kws)