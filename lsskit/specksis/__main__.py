"""
    __main__.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : analysis functions to be installed as console scripts
"""
import argparse
from lsskit.data import tools
from lsskit.specksis import io

def write_analysis_file():
    """
    Write a power spectrum to file as an analysis script
    """
    parser = argparse.ArgumentParser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # required arguments
    h = tools.PowerSpectraParser.format_help()
    parser.add_argument('data', type=tools.PowerSpectraParser.data, help=h)
    h = tools.PowerSpectraCallable.format_help()
    parser.add_argument('callable', type=tools.PowerSpectraCallable.data, help=h)
    h = 'the data keys to slice the data on'
    parser.add_argument('key', type=str, nargs='+', action=tools.StoreDataKeys, help=h)
    h = 'the data columns to write to the file'
    parser.add_argument('--cols', required=True, nargs='+', type=str)
    h = 'the output file name'
    parser.add_argument('-o', '--output', required=True, type=str, help=h)
    
    # options
    h = 'whether to keep missing data before writing; default = False'
    parser.add_argument('--keep_nans', action='store_true', default=False, help=h)
    h = 'whether to subtract the shot noise from the power before writing; default = False'
    parser.add_argument('--subtract_shot_noise', action='store_true', default=False, help=h)
    h = 'bins to reindex into, either in the `k` or `mu` dimensions'
    parser.add_argument('--reindex', nargs='*', action=tools.ReindexDict, default={}, help=h)
    
    # parse
    args = parser.parse_args()
    
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
    kwargs['remove_missing'] = not args.keep_nans
    kwargs['subtract_shot_noise'] = args.subtract_shot_noise
    kwargs['reindex'] = args.reindex
    io.write_analysis_file(args.output, power, args.cols, **kwargs)
    
    
    
    