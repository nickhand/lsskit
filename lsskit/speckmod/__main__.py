"""
    tools.py
    lsskit.speckmod
    
    __author__ : Nick Hand
    __desc__ : tools for helping with modeling of power spectra
"""
import fitit
from .. import numpy as np
from . import plugins

import argparse
import os

def perform_fit():
    """
    Fit a model to power spectrum data, installed as an entry point
    """
    # parse the input arguments
    parser = fitit.arg_parser.initialize_parser()
    parser.formatter_class = argparse.RawTextHelpFormatter

    # the input data  
    h = "the input data, specified as:\n\n"
    parser.add_argument('--data', required=True, type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('data'))
    
    # the input model  
    h = "the input model, specified as:\n\n"
    parser.add_argument('--model', required=True, type=plugins.ModelInput.parse,
                           help=h+plugins.ModelInput.format_help('model'))
                           
    
    # the outputs
    h = "any output storages, specified as:\n\n"
    parser.add_argument('--storage', '-X', action='append', 
                            type=plugins.ModelResultsStorage.parse,
                            help=h+plugins.ModelResultsStorage.format_help('output'))
    
    # parse the arguments
    args = fitit.arg_parser.parse_command_line(parser.parse_args())
    if args['subparser_name'] not in ['chisq', 'mcmc']:
        raise NotImplementedError("Only `chisq` and `mcmc` subparsers allowed; not %s" %args['subparser_name'])
    args['no_file_logger'] = True
    
    # get the parameters
    params, logfile = fitit.run.initialize_params(args)

    # get a few needed keys
    data = params['driver']['data'].value
    model = params['driver']['model'].value
    outputs = params['driver']['storage'].value
    mode = params['driver']['mode'].value
    
    # get the theory and model
    model, theory_params = fitit.store_extra_model_info(args['model'], params['theory'])

    # loop over the data
    for key, extra, data_df in data:
        
        key_str = '_'.join(map(str,key.values()))
        print "Fitting bin %s" %(", ".join("%s=%s" %(k,str(v)) for k,v in key.iteritems()))
        print "____________________________"

        # do the fit
        kwargs = {}
        if mode == 'chisq':
            result = fitit.driver.chisq_run(params['driver'], data_df, theory_params, model.copy(), **extra)
        elif mode == 'mcmc':
            result = fitit.driver.mcmc_run(params['driver'], data_df, theory_params, model.copy(), pool=None, **extra)
            kwargs['walkers'] = result.walkers
            kwargs['iterations'] = result.iterations
        else:
            raise NotImplementedError

        # summarize the fit
        if params['driver']['silent'].value:
            result.summarize_fit(to_screen=True)

        # save the output
        for output in outputs:
            output.write(dict(key, **extra), result)
        
    # now let's save the params too
    folder = params['driver']['folder'].value
    filename = os.path.join(folder, 'params.dat')
    with open(filename, 'w') as ff:
        for k,v in args.iteritems():
            if isinstance(v, list): v = map(str, v)
            ff.write("%s = %s\n" %(k, str(v)))
            
#------------------------------------------------------------------------------
def fit_gaussian_process():
    """
    Fit a Gaussian Process to either best-fit parameters or functions.
    This function is installed as an entry point
    """
    # parse the input arguments
    desc = "fit a Gaussian Process to either best-fit parameters or functions"
    parser = argparse.ArgumentParser(description=desc)
    parser.formatter_class = argparse.RawTextHelpFormatter

    # the gaussian process  
    h = "the input Gaussian Process arguments, specified as:\n\n"
    parser.add_argument('gp_args', type=plugins.ModelResultsStorage.parse, 
                            help=h+plugins.ModelResultsStorage.format_help('GP'))
                            
    args = parser.parse_args()
    args.gp_args.write()
    
#------------------------------------------------------------------------------
def compare_bestfit_functions():
    """
    Compare the best-fit model to the data; this function is installed as 
    an entry point
    """
    import plotify as pfy
    import pandas as pd
    
    # parse the input arguments
    desc = "compare the best-fit model to the data"
    parser = argparse.ArgumentParser(description=desc)
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # the input data  
    h = "the input data, specified as:\n\n"
    parser.add_argument('data', type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('data'))
                            
    # the bestfit function file
    h = 'the name of the file holding the dataframe of bestfit function means'
    parser.add_argument('bestfit_file', type=str, help=h)
        
    # the integer index to select an individual bin from
    h = 'the integer index to select an individual bin from'
    parser.add_argument('select', type=int, nargs='+', help=h)
    
    # parse
    args = parser.parse_args()
    
    # read the bestfits file and select
    df = pd.read_pickle(args.bestfit_file)
    if not all(x in df.columns for x in ['a', 'mass', 'k']):
        raise ValueError("please specify a bestfit file with `a`, `mass`, and `k` columns")
    df = df.set_index(['a', 'mass', 'k'])
    
    # get the key dictionary and print out what we are selecting
    key = dict((df.index.names[i], df.index.levels[i][v]) for i, v in enumerate(args.select))
    msg = ", ".join("%s = %s" %(k,v) for k,v in key.iteritems())
    print "selecting " + msg
    
    # select the bestfit
    select = tuple(df.index.levels[i][v] for i, v in enumerate(args.select))
    df = df.xs(select)
    
    # select the data
    data_df = args.data.to_dataframe(key)
    
    # plot the data
    pfy.errorbar(data_df.index.values, data_df['y'], data_df['error'])
    
    # plot the bestfit function mean
    x = df.index.values
    y = df['mean']
    errs = df['error']
    lines = pfy.plot(x, y)
    pfy.plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([y - errs,
                                (y + errs)[::-1]]),
                                alpha=.5, fc=lines[0].get_color(), ec='None')
                                
    ax = pfy.gca()
    ax.title.update('Bestfit comparison for %s' %msg, fontsize=16)
    ax.xlabel.update(r"$k$ ($h$/Mpc)", fontsize=16)
    ax.ylabel.update(args.data.variable_str, fontsize=16)
    pfy.show()

#------------------------------------------------------------------------------

    
    
    
    
    

