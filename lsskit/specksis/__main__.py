"""
    __main__.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : analysis functions to be installed as console scripts
"""
import argparse
from .. import numpy as np
from ..data import parse_tools
from ..specksis import io, tools, covariance


#------------------------------------------------------------------------------
# writing analysis/grid files
#------------------------------------------------------------------------------
def write_analysis_file():
    """
    Write a P(k,mu), P(k) or multipoles spectrum as an 'analysis' 
    plaintext file
    """
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # required arguments
    h = 'the kind of analysis file to write, either `power` or `poles`'
    parser.add_argument('kind', choices=['power', 'poles'], help=h)
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
    parser.add_argument('--kmin', nargs='+', type=float, default=[-np.inf], help=h)
    h = "the maximum wavenumber to use"
    parser.add_argument('--kmax', nargs='+', type=float, default=[np.inf], help=h)
    
    # parse
    args = parser.parse_args()
    if len(args.kmin) == 1:
        args.kmin = args.kmin[0]
    if len(args.kmax) == 1:
        args.kmax = args.kmax[0]
    
    # get the data from the parent data and function
    data = getattr(args.data, args.callable['name'])(**args.callable['kwargs'])
    coords = data.coords
    for c in coords:
        if c in args.key:
            dt = data.coords[c].dtype.type
            args.key[c] = [dt(x) for x in args.key[c]]
    
    # now slice
    for k in args.key:
        if len(args.key[k]) != 1:
            raise ValueError("must specify exactly one key for each dimension")
        args.key[k] = args.key[k][0]
        
    for k in args.key:
        cast = data[k].dtype.type
        args.key[k] = cast(args.key[k])
    try:
        power = data.sel(**args.key)
        if power.size == 1: power = power.get()
    except Exception as e:
        raise RuntimeError("error slicing data with key %s: %s" %(str(args.key), str(e)))

    # now output
    kwargs = {}
    kwargs['subtract_shot_noise'] = args.subtract_shot_noise
    kwargs['kmin'] = args.kmin
    kwargs['kmax'] = args.kmax
    io.write_analysis_file(args.kind, args.output, power, args.cols, **kwargs)

def write_analysis_grid():
    """
    Write a P(k,mu) grid for analysis purposes
    """
    from pyRSD.rsd import PkmuGrid
    
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # required arguments
    h = parse_tools.PowerSpectraParser.format_help()
    parser.add_argument('data', type=parse_tools.PowerSpectraParser.data, help=h)
    h = parse_tools.PowerSpectraCallable.format_help()
    parser.add_argument('callable', type=parse_tools.PowerSpectraCallable.data, help=h)
    h = "the (optional) data keys to slice the data on; specified as `a = '0.6452'`"
    parser.add_argument('--key', type=str, nargs='+', action=parse_tools.StoreDataKeys, help=h)
    h = 'the output file name'
    parser.add_argument('-o', '--output', required=True, type=str, help=h)
    args = parser.parse_args()
    
    # get the data from the parent data and function
    data = getattr(args.data, args.callable['name'])(**args.callable['kwargs'])
    
    if args.key is not None:
        coords = data.coords
        for c in coords:
            if c in args.key:
                dt = data.coords[c].dtype.type
                args.key[c] = [dt(x) for x in args.key[c]]
            
    # now slice
    if args.key is not None:
        for k in args.key:
            if len(args.key[k]) != 1:
                raise ValueError("must specify exactly one key for each dimension")
            args.key[k] = args.key[k][0]
        try:
            data = data.sel(**args.key)
            if data.size == 1: data = data.get()
        except Exception as e:
            raise RuntimeError("error slicing data with key %s: %s" %(str(args.key), str(e)))
    
    # now make the grid and save to plaintext file
    grid = PkmuGrid.from_pkmuresult(data)
    grid.to_plaintext(args.output)


#------------------------------------------------------------------------------
# writing covariance matrices
#------------------------------------------------------------------------------
def write_covariance():
    """
    Write out a P(k,mu) or multipoles covariance matrix
    """
    from pyRSD.rsdfit.data import PkmuCovarianceMatrix, PoleCovarianceMatrix
    
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
    parser.add_argument('--ells', nargs='*', type=int, help=h)
    
    # options
    h = "the minimum wavenumber to use"
    parser.add_argument('--kmin', nargs='+', type=float, default=[-np.inf], help=h)
    h = "the maximum wavenumber to use"
    parser.add_argument('--kmax', nargs='+', type=float, default=[np.inf], help=h)
    h = "set off-diagonal elements to zero"
    parser.add_argument('--force_diagonal', action='store_true', help=h)
        
    # parse
    args = parser.parse_args()
    if len(args.kmin) == 1: args.kmin = args.kmin[0]
    if len(args.kmax) == 1: args.kmax = args.kmax[0]
    
    # get the data from the parent data and function
    data = getattr(args.data, args.callable['name'])(**args.callable['kwargs'])
    
    # compute the covariance matrix
    kwargs = {}
    kwargs['kmin'] = args.kmin
    kwargs['kmax'] = args.kmax
    kwargs['force_diagonal'] = args.force_diagonal
    
    if args.mode == 'pkmu':
        C = PkmuCovarianceMatrix.from_spectra_set(data, **kwargs)    
    elif args.mode == 'poles':
        if args.ells is not None: kwargs['ells'] = args.ells
        C = PoleCovarianceMatrix.from_spectra_set(data, **kwargs)
    
    # now output
    if args.format == 'pickle':
        C.to_pickle(args.output)
    else:
        C.to_plaintext(args.output)
        
def write_data_gaussian_covariance():
    """
    Write out the gaussian covariance matrix from data measurements, 
    for either P(k,mu) or multipoles
    """
    from pyRSD.rsdfit.data import PkmuCovarianceMatrix, PoleCovarianceMatrix
    
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    subparsers = parser.add_subparsers(dest='subparser_name')
    
    # pkmu parser
    pkmu_parser = subparsers.add_parser('pkmu')
    h = parse_tools.PowerSpectraParser.format_help()
    pkmu_parser.add_argument('data', type=parse_tools.PowerSpectraParser.data, help=h)
    h = parse_tools.PowerSpectraCallable.format_help()
    pkmu_parser.add_argument('callable', type=parse_tools.PowerSpectraCallable.data, help=h)
    h = 'a list of (lower, upper) specifying the mu bin bounds'
    pkmu_parser.add_argument('mu_bounds', type=str, help=h)
    h = "the data keys to slice the data on; specified as `a = '0.6452'`"
    pkmu_parser.add_argument('key', type=str, nargs='+', action=parse_tools.StoreDataKeys, help=h)
    
    # poles parser
    pole_parser = subparsers.add_parser('poles')
    h = parse_tools.PowerSpectraParser.format_help()
    pole_parser.add_argument('data', type=parse_tools.PowerSpectraParser.data, help=h)
    h = parse_tools.PowerSpectraCallable.format_help()
    pole_parser.add_argument('callable', type=parse_tools.PowerSpectraCallable.data, help=h)
    h = 'the multipole numbers'
    pole_parser.add_argument('ells', type=str, help=h)
    h = "the data keys to slice the data on; specified as `a = '0.6452'`"
    pole_parser.add_argument('key', type=str, nargs='+', action=parse_tools.StoreDataKeys, help=h)

    # options
    for p in [pkmu_parser, pole_parser]:
        h = 'the data format to use, either `pickle` or `plaintext`'
        p.add_argument('--format', choices=['pickle', 'plaintext'], default='plaintext', help=h)
        h = 'the output file name'
        p.add_argument('-o', '--output', required=True, type=str, help=h)
        h = "the minimum wavenumber to use"
        p.add_argument('--kmin', nargs='+', type=float, default=[-np.inf], help=h)
        h = "the maximum wavenumber to use"
        p.add_argument('--kmax', nargs='+', type=float, default=[np.inf], help=h)
        
    # parse
    args = parser.parse_args()
    if len(args.kmin) == 1: args.kmin = args.kmin[0]
    if len(args.kmax) == 1: args.kmax = args.kmax[0]
    
    # get the data from the parent data and function
    data = getattr(args.data, args.callable['name'])(**args.callable['kwargs'])
    coords = data.coords
    for c in coords:
        if c in args.key:
            dt = data.coords[c].dtype.type
            args.key[c] = [dt(x) for x in args.key[c]]
    
    # now slice
    for k in args.key:
        if len(args.key[k]) != 1:
            raise ValueError("must specify exactly one key for each dimension")
        args.key[k] = args.key[k][0]
    try:
        data = data.sel(**args.key)
        if data.size == 1: data = data.get()
    except Exception as e:
        raise RuntimeError("error slicing data with key %s: %s" %(str(args.key), str(e)))
    
    # compute the covariance matrix
    if args.subparser_name == 'pkmu':
        mu_bounds = eval(args.mu_bounds)
        C, coords = covariance.data_pkmu_gausscov(data, mu_bounds, kmin=args.kmin, kmax=args.kmax)
        C = PkmuCovarianceMatrix(C, coords[0], coords[1])
    else:
        ells = np.array(eval(args.ells), dtype=float)
        C, coords = covariance.data_pole_gausscov(data, ells, kmin=args.kmin, kmax=args.kmax)
        C = PoleCovarianceMatrix(C, coords[0], coords[1])
    
    # now output
    if args.format == 'pickle':
        C.to_pickle(args.output)
    else:
        C.to_plaintext(args.output)
        
#------------------------------------------------------------------------------
# MCMC helper scripts
#------------------------------------------------------------------------------        
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
        print(bestfit.to_comparison_table(data, args.output, **kwargs))
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
            key_str = " ".join(["%s = %s" %(k, str(v)) for k,v in key.items()])
            print("rank %d: processing %s ..." %(rank, key_str))
        
            # compute the multipoles
            spec = spec.get()
            poles = spec.to_multipoles(*args.ell)
        
            valid = {k:v for k,v in key.items() if k in str_kwargs}
            for iell , ell in enumerate(args.ell):
                valid['pole'] = pole_names[ell]
                filename = args.output.format(**valid)
                poles[iell].to_plaintext(filename)
            
        
    
    
    
    