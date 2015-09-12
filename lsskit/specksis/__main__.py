"""
    __main__.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : analysis functions to be installed as console scripts
"""
import argparse
from ..data import parse_tools
from ..specksis import io, tools

def write_power_analysis_file():
    """
    Write a power spectrum to file as an analysis script
    """
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # required arguments
    h = parse_tools.PowerSpectraParser.format_help()
    parser.add_argument('data', type=parse_tools.PowerSpectraParser.data, help=h)
    h = parse_tools.PowerSpectraCallable.format_help()
    parser.add_argument('callable', type=parse_tools.PowerSpectraCallable.data, help=h)
    h = "the data keys to slice the data on; specified as `a = '0.6452'`"
    parser.add_argument('key', type=str, nargs='+', action=parse_tools.StoreDataKeys, help=h)
    h = 'the data columns to write to the file'
    parser.add_argument('--cols', required=True, nargs='+', type=str)
    h = 'the output file name'
    parser.add_argument('-o', '--output', required=True, type=str, help=h)
    
    # options
    h = 'whether to subtract the shot noise from the power before writing; default = False'
    parser.add_argument('--subtract_shot_noise', action='store_true', default=False, help=h)
    h = "the minimum wavenumber to use"
    parser.add_argument('--kmin', nargs='+', type=float, default=None, help=h)
    h = "the maximum wavenumber to use"
    parser.add_argument('--kmax', nargs='+', type=float, default=None, help=h)
    
    # parse
    args = parser.parse_args()
    if args.kmin is not None and len(args.kmin) == 1:
        args.kmin = args.kmin[0]
    if args.kmax is not None and len(args.kmax) == 1:
        args.kmax = args.kmax[0]
    
    # get the data from the parent data and function
    data = getattr(args.data, args.callable['name'])(**args.callable['kwargs'])
    
    # now slice
    for k in args.key:
        if len(args.key[k]) != 1:
            raise ValueError("must specify exactly one key for each dimension")
        args.key[k] = args.key[k][0]
    try:
        power = data.sel(**args.key)
        if power.size == 1:
            power = power.values
    except Exception as e:
        raise RuntimeError("error slicing data with key %s: %s" %(str(args.key), str(e)))

    # now output
    kwargs = {}
    kwargs['subtract_shot_noise'] = args.subtract_shot_noise
    kwargs['kmin'] = args.kmin
    kwargs['kmax'] = args.kmax
    io.write_power_analysis_file(args.output, power, args.cols, **kwargs)
    
    
def write_poles_analysis_file():
    """
    Write multipoles to file as an analysis script
    """
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # required arguments
    h = parse_tools.PowerSpectraParser.format_help()
    parser.add_argument('data', type=parse_tools.PowerSpectraParser.data, help=h)
    h = parse_tools.PowerSpectraCallable.format_help()
    parser.add_argument('poles_callable', type=parse_tools.PowerSpectraCallable.data, help=h)
    h = parse_tools.PowerSpectraCallable.format_help()
    parser.add_argument('pkmu_callable', type=parse_tools.PowerSpectraCallable.data, help=h)
    h = "the data keys to slice the data on; specified as `a = '0.6452'`"
    parser.add_argument('key', type=str, nargs='+', action=parse_tools.StoreDataKeys, help=h)
    h = 'the data columns to write to the file'
    parser.add_argument('--cols', required=True, nargs='+', type=str)
    h = 'the output file name'
    parser.add_argument('-o', '--output', required=True, type=str, help=h)
    
    # options
    h = 'whether to subtract the shot noise from the power before writing; default = False'
    parser.add_argument('--subtract_shot_noise', action='store_true', default=False, help=h)
    h = "the minimum wavenumber to use"
    parser.add_argument('--kmin', nargs='+', type=float, default=None, help=h)
    h = "the maximum wavenumber to use"
    parser.add_argument('--kmax', nargs='+', type=float, default=None, help=h)
    
    # parse
    args = parser.parse_args()
    if args.kmin is not None and len(args.kmin) == 1:
        args.kmin = args.kmin[0]
    if args.kmax is not None and len(args.kmax) == 1:
        args.kmax = args.kmax[0]
    
    # get the data from the parent data and function
    data = getattr(args.data, args.poles_callable['name'])(**args.poles_callable['kwargs'])
    pkmu = getattr(args.data, args.pkmu_callable['name'])(**args.pkmu_callable['kwargs'])
    
    # now slice
    for k in args.key:
        if len(args.key[k]) != 1:
            raise ValueError("must specify exactly one key for each dimension")
        args.key[k] = args.key[k][0]
    try:
        poles = data.sel(**args.key)
        pkmu = pkmu.sel(**args.key).values
    except Exception as e:
        raise RuntimeError("error slicing data with key %s: %s" %(str(args.key), str(e)))

    # now output
    kwargs = {}
    kwargs['subtract_shot_noise'] = args.subtract_shot_noise
    kwargs['kmin'] = args.kmin
    kwargs['kmax'] = args.kmax
    io.write_poles_analysis_file(args.output, poles, pkmu, args.cols, **kwargs)

def write_covariance():
    """
    Write out a power spectrum covariance matrix
    """
    from pyRSD.rsdfit.data import CovarianceMatrix
    
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # required arguments
    h = 'the mode, either `pkmu` or `poles`'
    parser.add_argument('mode', choices=['pkmu', 'poles'], help=h)
    h = parse_tools.PowerSpectraParser.format_help()
    parser.add_argument('data', type=parse_tools.PowerSpectraParser.data, help=h)
    h = parse_tools.PowerSpectraCallable.format_help()
    parser.add_argument('callable', type=parse_tools.PowerSpectraCallable.data, help=h)
    h = 'the data format to use, either `pickle` or `plaintext`'
    parser.add_argument('--format', choices=['pickle', 'plaintext'], default='plaintext', help=h)
    h = 'the output file name'
    parser.add_argument('-o', '--output', required=True, type=str, help=h)
    h = 'the multipole numbers to compute'
    parser.add_argument('-l', '--ell', nargs='*', type=int, help=h)
    
    # options
    h = "the minimum wavenumber to use"
    parser.add_argument('--kmin', nargs='+', type=float, default=None, help=h)
    h = "the maximum wavenumber to use"
    parser.add_argument('--kmax', nargs='+', type=float, default=None, help=h)
    h = "set off-diagonal elements to zero"
    parser.add_argument('--force_diagonal', action='store_true', help=h)
        
    # parse
    args = parser.parse_args()
    if args.kmin is not None and len(args.kmin) == 1:
        args.kmin = args.kmin[0]
    if args.kmax is not None and len(args.kmax) == 1:
        args.kmax = args.kmax[0]
    
    # get the data from the parent data and function
    data = getattr(args.data, args.callable['name'])(**args.callable['kwargs'])
    
    # compute the covariance matrix
    kwargs = {}
    if args.kmin is not None: kwargs['kmin'] = args.kmin
    if args.kmax is not None: kwargs['kmax'] = args.kmax
    kwargs['force_diagonal'] = args.force_diagonal
    kwargs['return_extras'] = False
    if args.mode == 'pkmu':
        C = tools.compute_pkmu_covariance(data, **kwargs)    
    elif args.mode == 'poles':
        if args.ell is None:
            raise ValueError("if `mode = poles`, then `ell` must be supplied")
        C = tools.compute_pole_covariance(data, args.ell, **kwargs)
    
    # now output
    C = CovarianceMatrix(C, verify=False)
    if args.format == 'pickle':
        C.to_pickle(args.output)
    else:
        C.to_plaintext(args.output)
        
def plot_mcmc_bestfit():
    """
    Load a mcmc bestfit and plot it
    """
    from pyRSD.rsdfit import FittingDriver
    import plotify as pfy
    
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # arguments
    h = 'the name of directory holding the results'
    parser.add_argument('results_dir', type=str, help=h)
    h = 'the name of the results file to load'
    parser.add_argument('results_file', type=str, help=h)
    h = 'the name of the model to load'
    parser.add_argument('model', type=str, help=h)
    h = 'the output file name'
    parser.add_argument('-o', '--output', type=str, help=h)
    h = 'the number of burnin steps to set'
    parser.add_argument('-b', '--burnin', type=int, help=h)
    
    args = parser.parse_args()
    
    # load the driver
    kwargs = {'results_file':args.results_file, 'model_file':args.model}
    driver = FittingDriver.from_directory(args.results_dir, **kwargs)
    
    if args.burnin is not None:
        driver.results.burnin = args.burnin
    
    driver.set_fit_results()
    driver.plot()
    
    if args.output is not None:
        pfy.savefig(args.output)
    pfy.show()
    
def compare_mcmc_fits():
    """
    Write a latex table comparing multiple mcmc fits
    """
    from pyRSD.rsdfit.analysis import bestfit
    
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # arguments
    h = 'the names of the files to load the fits from'
    parser.add_argument('--files', required=True, nargs='+', type=str, help=h)
    h = 'the names of each fit'
    parser.add_argument('--names', required=True, nargs='+', type=str, help=h)
    h = 'the output file name'
    parser.add_argument('-o', '--output', type=str, help=h)
    h = 'convenience option to only include the free parameters'
    parser.add_argument('--free-only', action='store_true', help=h)
    h = 'convenience option to only include the constrained parameters'
    parser.add_argument('--constrained-only', action='store_true', help=h)
    h = 'only include these parameters in the output table'
    parser.add_argument('--params', nargs='*', help=h, type=str)
    
    args = parser.parse_args()
    data = {}
    for name, f in zip(args.names, args.files):
        data[name] = bestfit.BestfitParameterSet.from_info(f)
        
    kwargs = {'free_only':args.free_only, 'constrained_only':args.constrained_only, 'params':args.params}
    if args.output is None:
        print bestfit.to_comparison_table(data, args.output, **kwargs)
    else:
        bestfit.to_comparison_table(data, args.output, **kwargs)
        
def compute_multipoles():
    """
    Compute the power spectrum multipoles from a set of P(k, mu) measurements
    """
    from nbodykit import pkmuresult
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
                    
    # parse the input arguments
    desc = "compute the power spectrum multipoles from a set of P(k, mu) measurements"
    parser = argparse.ArgumentParser(description=desc)
    
    # required arguments
    h = parse_tools.PowerSpectraParser.format_help()
    parser.add_argument('data', type=parse_tools.PowerSpectraParser.data, help=h)
    h = parse_tools.PowerSpectraCallable.format_help()
    parser.add_argument('callable', type=parse_tools.PowerSpectraCallable.data, help=h)
    h = 'the name of the output file'
    parser.add_argument('-o', '--output', type=str, required=True, help=h)
    h = 'the multipole numbers to compute'
    parser.add_argument('-l', '--ell', nargs='+', type=int, help=h)
    
    args = parser.parse_args()
    
    # load the Pkmu spectra
    pkmu = getattr(args.data, args.callable['name'])(**args.callable['kwargs'])
    
    # output string kwargs
    str_kwargs = [kw for _, kw, _, _ in args.output._formatter_parser() if kw]
    if 'pole' not in str_kwargs:
        raise ValueError("`pole` must be a string format keyword in `output` for naming")
    pole_names = {0:'mono', 2:'quad', 4:'hexadec'}
    if any(ell not in pole_names.keys() for ell in args.ell):
        raise ValueError("valid `ell` integers are %s" %str(pole_names.keys()))
    
    if isinstance(pkmu, pkmuresult.PkmuResult):
        poles = pkmu.to_multipoles(*args.ell)
        for iell , ell in enumerate(args.ell):
            filename = args.output.format(pole=pole_names[ell])
            poles[iell].to_plaintext(filename)
    else:    
        for i, (key, spec) in enumerate(pkmu.nditer()):
            if i % size != rank:
                continue    
            key_str = " ".join(["%s = %s" %(k, str(v)) for k,v in key.iteritems()])
            print "rank %d: processing %s ..." %(rank, key_str)
        
            # compute the multipoles
            spec = spec.values
            poles = spec.to_multipoles(*args.ell)
        
            valid = {k:v for k,v in key.iteritems() if k in str_kwargs}
            for iell , ell in enumerate(args.ell):
                valid['pole'] = pole_names[ell]
                filename = args.output.format(**valid)
                poles[iell].to_plaintext(filename)
            
        
    
    
    
    