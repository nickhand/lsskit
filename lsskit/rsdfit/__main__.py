"""
    __main__.py
    lsskit.rsdfit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : module to hold the top-level console scripts for rsdfit
"""
import argparse 
import os
import sys

from lsskit.rsdfit import lib, sync
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
    
    h = 'the number of nodes to use when submitting the job'
    parser.add_argument('-N', '--nodes', type=int, help=h)
    
    h = 'the partition to submit the job to'
    parser.add_argument('-p', '--partition', type=str, choices=['debug', 'regular'], help=h)
    
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

    # get the kwargs
    kws = {}
    kws['rsdfit_options'] = other
    kws['theory_options'] = ns.theory_options
    kws['tag'] = ns.tag
    kws['command'] = ns.command
    kws['nodes'] = ns.nodes
    kws['partition'] = ns.partition

    # can accept input ``box`` values to loop over
    if not sys.stdin.isatty():
        
        # the iteration values to loop over
        itervalues = [line.split() for line in sys.stdin.readlines()]
        iterkeys = itervalues[0]
        itervalues = itervalues[1:]
        
        # the input config file
        config = open(ns.config, 'r').read()
        
        # run for each iteration
        for ival in itervalues:
           
            # make the param file and run
            try:
                param_file = lib.make_temp_config(config, iterkeys, ival)
                lib.run_rsdfit(param_file, ns.stat, ns.kmax, **kws)
            
            # if KeyboardInterrupt, continue on to next
            except KeyboardInterrupt:
                pass
            except Exception as e:
                raise e
            finally:
                # if we actually ran rsdfit, delete the temp file
                if ns.partition is None and ns.nodes is None:
                    if os.path.exists(param_file):
                       os.remove(param_file)

    # just call the ``rsdfit`` command
    else: # no stdin data
        lib.run_rsdfit(ns.config, ns.stat, ns.kmax, **kws)
        

def iter_rsdfit():
    """
    Iterate over one or more sequences, printing out the product
    
    Examples
    --------
    >> iter_rsdfit "box: [1, 2, 3]; los: ['x', 'y', 'z']"
    box los
    1 x
    1 y
    1 z
    2 x
    2 y
    2 z
    3 x
    3 y
    3 z
    """
    import yaml
    import itertools
    
    desc = "iterate over one or more sequences, print the product to stdout"
    parser = argparse.ArgumentParser(description=desc)
    
    h = "the input list of sequences, of the form key1: seq1; key2: seq2"
    parser.add_argument('input', type=str, help=h)
    
    ns = parser.parse_args()
        
    # do the yaml load, replacing any semicolons with newlines
    s = ns.input.replace(';', '\n')
    s = s.replace('\n ', '\n')
    s = yaml.load(s)
    for k in s:
        if not isinstance(s[k], list):
            s[k] = eval(s[k])
    
    # take the product
    keys = list(s.keys())
    output = itertools.product(*[s[k] for k in keys])
    
    # make the individual lines of the output
    toret = [" ".join(keys) + '\n']
    for v in output:
        toret.append(" ".join(map(str, v)) + '\n')
    
    # write to stdout
    sys.stdout.writelines(toret)
    
def sync_rsdfit():
    """
    Sync the ``rsdfit`` models, using `rsync`
    """
    # the main parser
    desc = "sync the ``rsdfit`` directory structure using `rsync`"
    parser = argparse.ArgumentParser(description=desc)
    subparsers = parser.add_subparsers(dest='subparser_name')
    
    # setup the parent
    parent = argparse.ArgumentParser(add_help=False)
    h = 'the remote host; either `cori` or `edison`'
    parent.add_argument('host', type=str, choices=['cori', 'edison'], help=h)
    h = 'show what would have been transferred'
    parent.add_argument('-n', '--dry-run', action='store_true', help=h)
    
    # models
    h = 'sync the `$RSDFIT_MODELS` directory'
    models = subparsers.add_parser('models', parents=[parent], help=h)
    models.set_defaults(func=sync.sync_models)
    
    # data
    h = 'sync the `$RSDFIT_DATA` directory'
    data = subparsers.add_parser('data', parents=[parent], help=h)
    data.set_defaults(func=sync.sync_data)
    
    # params
    h = 'sync the `$RSDFIT/params` directory'
    data = subparsers.add_parser('params', parents=[parent], help=h)
    data.set_defaults(func=sync.sync_params)
    
    # run
    h = 'sync the `$RSDFIT/run` directory'
    data = subparsers.add_parser('run', parents=[parent], help=h)
    data.set_defaults(func=sync.sync_run)

    ns = parser.parse_args()
    ns.func(ns.host, dry_run=ns.dry_run)
    
def sync_rsdfit_fits():
    """
    Sync the ``rsdfit`` fits, using `rsync`
    """
    desc = "sync the ``rsdfit`` fits stored in ``RSDFIT_FITS``, using `rsync`"
    parser = argparse.ArgumentParser(description=desc)

    h = 'the transfer direction; either `to` or `from` the remote host'
    parser.add_argument('direction', type=str, choices=['to', 'from'], help=h)
    
    h = 'the remote host; either `cori` or `edison`'
    parser.add_argument('host', type=str, choices=['cori', 'edison'], help=h)
    
    h = 'an additional subpath from RSDFIT_FITS to sync only'
    parser.add_argument('-d', '--dir', type=str, help=h)

    h = 'show what would have been transferred'
    parser.add_argument('-n', '--dry-run', action='store_true', help=h)
    
    ns = parser.parse_args()
    sync.sync_fits(ns.direction, ns.host, path=ns.dir, dry_run=ns.dry_run)
    
    