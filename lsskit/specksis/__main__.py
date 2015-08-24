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

def write_analysis_file():
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
        power = data.sel(**args.key).values
    except Exception as e:
        raise RuntimeError("error slicing data with key %s: %s" %(str(args.key), str(e)))
        
    # now output
    kwargs = {}
    kwargs['subtract_shot_noise'] = args.subtract_shot_noise
    kwargs['kmin'] = args.kmin
    kwargs['kmax'] = args.kmax
    io.write_analysis_file(args.output, power, args.cols, **kwargs)

def write_covariance():
    """
    Write out a power spectrum covariance matrix
    """
    from pyRSD.rsdfit.data import CovarianceMatrix
    
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # required arguments
    h = parse_tools.PowerSpectraParser.format_help()
    parser.add_argument('data', type=parse_tools.PowerSpectraParser.data, help=h)
    h = parse_tools.PowerSpectraCallable.format_help()
    parser.add_argument('callable', type=parse_tools.PowerSpectraCallable.data, help=h)
    h = 'the data format to use, either `pickle` or `plaintext`'
    parser.add_argument('--format', choices=['pickle', 'plaintext'], default='plaintext', help=h)
    h = 'the output file name'
    parser.add_argument('-o', '--output', required=True, type=str, help=h)
    
    # options
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
    
    # compute the covariance matrix
    kwargs = {}
    if args.kmin is not None: kwargs['kmin'] = args.kmin
    if args.kmax is not None: kwargs['kmax'] = args.kmax
    _, _, _, C = tools.compute_pkmu_covariance(data, **kwargs)    
    
    # now output
    C = CovarianceMatrix(C, verify=False)
    if args.format == 'pickle':
        C.to_pickle(args.output)
    else:
        C.to_plaintext(args.output)
    
    
    
    