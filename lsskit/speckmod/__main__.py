"""
    tools.py
    lsskit.speckmod
    
    __author__ : Nick Hand
    __desc__ : tools for helping with modeling of power spectra
"""
import fitit
from .. import numpy as np
from . import plugins, tools

import argparse
import os

def perform_fit():
    """
    Fit a model to power spectrum data, installed as an entry point
    """
    # parse the input arguments
    parser = fitit.arg_parser.initialize_parser()
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.fromfile_prefix_chars = '@'
    parser.convert_arg_line_to_args = lambda line: tools.convert_arg_line_to_args(parser, line)

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
    folder = params['driver']['folder'].value
    
    # prepend the output folder to save results
    for output in outputs:
        output_name = os.path.join(folder, output.path)
        odir = os.path.dirname(output_name)
        if not os.path.isdir(odir):
            raise RuntimeError("output directory `%s` must exist" %odir)
        output.path = output_name
    
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
                            help=h+plugins.ModelResultsStorage.format_help('gp_interp'))
                            
    args = parser.parse_args()
    args.gp_args.write()
    
#------------------------------------------------------------------------------
def fit_spline_table():
    """
    Fit a spline interpolation table to the best-fit parameters.
    This function is installed as an entry point
    """
    # parse the input arguments
    desc = "fit a spline interpolation table to the best-fit parameters"
    parser = argparse.ArgumentParser(description=desc)
    parser.formatter_class = argparse.RawTextHelpFormatter

    # the gaussian process  
    h = "the input spline table arguments, specified as:\n\n"
    parser.add_argument('spline_args', type=plugins.ModelResultsStorage.parse, 
                            help=h+plugins.ModelResultsStorage.format_help('spline_interp'))
                            
    args = parser.parse_args()
    args.spline_args.write()
    
#------------------------------------------------------------------------------   
def compare():
    """
    Compare the best-fit model to the data.
    """
    # parse the input arguments
    desc = "compare the best-fit model to the data"
    parser = argparse.ArgumentParser(description=desc)
    parser.formatter_class = argparse.RawTextHelpFormatter
    subparsers = parser.add_subparsers(dest='subparser_name', help='sub-command help')
    
    #--------------------------------------------------------------
    # Function parser
    #--------------------------------------------------------------
    h = "compare the best-fit function to the data"
    func_parser = subparsers.add_parser('function', help=h)
    func_parser.formatter_class = argparse.RawTextHelpFormatter
    
    # the input data  
    h = "the input data, specified as:\n\n"
    func_parser.add_argument('data', type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('data'))
                            
    # the bestfit function file
    h = 'the name of the file holding the dataframe of bestfit parameters'
    func_parser.add_argument('bestfit_file', type=str, help=h)
        
    # the integer index to select an individual bin from
    h = 'the integer index to select an individual bin from;'
    h += ' the format should be `index_col`:value'
    func_parser.add_argument('select', type=str, nargs='+', help=h)
    
    #--------------------------------------------------------------
    # Params parser
    #--------------------------------------------------------------
    h = "compare the best-fit parameters to the data"
    param_parser = subparsers.add_parser('params', help=h)
    param_parser.formatter_class = argparse.RawTextHelpFormatter
    
    # the input data  
    h = "the input data, specified as:\n\n"
    param_parser.add_argument('data', type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('data'))
                            
    # the input model  
    h = "the input model, specified as:\n\n"
    param_parser.add_argument('model', type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('model'))
                            
    # the bestfit function file
    h = 'the name of the file holding the dataframe of bestfit parameters'
    param_parser.add_argument('bestfit_file', type=str, help=h)
        
    # the integer index to select an individual bin from
    h = 'the integer index to select an individual bin from;'
    h += ' the format should be `index_col`:value'
    param_parser.add_argument('select', type=str, nargs='+', help=h)
    
    #--------------------------------------------------------------
    # GP parser
    #--------------------------------------------------------------
    h = "compare the best-fit parameters from a GP to the data"
    gp_parser = subparsers.add_parser('gp', help=h)
    gp_parser.formatter_class = argparse.RawTextHelpFormatter
    
    # the input data  
    h = "the input data, specified as:\n\n"
    gp_parser.add_argument('data', type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('data'))
                            
    # the input model  
    h = "the input model, specified as:\n\n"
    gp_parser.add_argument('model', type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('model'))
                            
    # the bestfit function file
    h = 'the name of the file holding the dataframe of bestfit parameters'
    gp_parser.add_argument('bestfit_file', type=str, help=h)
    
    # the gp model file
    h = 'the name of the file holding the GP'
    gp_parser.add_argument('gp_file', type=str, help=h)
        
    # the integer index to select an individual bin from
    h = 'the integer index to select an individual bin from;'
    h += ' the format should be `index_col`:value'
    gp_parser.add_argument('select', type=str, nargs='+', help=h)
    
    #--------------------------------------------------------------
    # spline table parser
    #--------------------------------------------------------------
    h = "compare the best-fit parameters from a spline table to the data"
    spline_parser = subparsers.add_parser('spline', help=h)
    spline_parser.formatter_class = argparse.RawTextHelpFormatter
    
    # the input data  
    h = "the input data, specified as:\n\n"
    spline_parser.add_argument('data', type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('data'))
                            
    # the input model  
    h = "the input model, specified as:\n\n"
    spline_parser.add_argument('model', type=plugins.ModelInput.parse, 
                            help=h+plugins.ModelInput.format_help('model'))
                            
    # the bestfit function file
    h = 'the name of the file holding the dataframe of bestfit parameters'
    spline_parser.add_argument('bestfit_file', type=str, help=h)
    
    # the gp model file
    h = 'the name of the file holding the spline table'
    spline_parser.add_argument('spline_file', type=str, help=h)
        
    # the integer index to select an individual bin from
    h = 'the integer index to select an individual bin from;'
    h += ' the format should be `index_col`:value'
    spline_parser.add_argument('select', type=str, nargs='+', help=h)
    
    # parse
    args = parser.parse_args()
    
    if args.subparser_name == 'function':
        tools.compare_bestfits('function', **vars(args))
    elif args.subparser_name == 'params':
        tools.compare_bestfits('params', **vars(args))
    elif args.subparser_name == 'gp':
        tools.compare_bestfits('gp', **vars(args))
    elif args.subparser_name == 'spline':
        tools.compare_bestfits('spline', **vars(args))

#------------------------------------------------------------------------------  
def add_bestfit_param():
    """
    Add a new parameter to a bestfit parameter file
    """
    import pandas as pd
    
    # parse the input arguments
    desc = "add a new parameter to a bestfit parameter file"
    parser = argparse.ArgumentParser(description=desc)
    parser.formatter_class = argparse.RawTextHelpFormatter
    
    # the bestfit function file
    h = 'the name of the file holding the dataframe of bestfit parameters'
    parser.add_argument('bestfit_file', type=str, help=h)
    
    # the expression
    h = 'the expression for the new parameter; should be of the form'
    h += ' param_name = expr'
    parser.add_argument('expr', type=str, help=h)
    
    # the error type
    h = 'the type of error calculation'
    parser.add_argument('error', type=str, choices=['absolute', 'fractional'], help=h)
    
    # the expression
    h = 'the name of the output file'
    parser.add_argument('-o', '--output', type=str, help=h)
    
    args = parser.parse_args()
    tools.add_bestfit_param(**vars(args))

    
    
    

