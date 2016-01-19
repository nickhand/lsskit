import argparse

def add_model_cache(parser):
    """
    Adds the necessary parameter options to handle
    caching of models
    """
    h = 'the RSD model path to load'
    parser.add_argument('--model_path', type=str, help=h)
    
    h = 'save the RSD model to the file specified by `model_path`'
    parser.add_argument('--cache-model', action='store_true', help=h)
    
    h = 'ignore any cached models and initialize a new RSD model'
    parser.add_argument('--ignore-cache', action='store_true', help=h)

def parse_binned_mcmc(desc, dims, coords, return_parser=False, cache_model=True):
    """
    Return the parsed arguments for a set of bins that we will 
    individually run an mcmc on
    """
    parser = argparse.ArgumentParser(description=desc)
    
    h = 'the name of the mcmc parameter file to load'
    parser.add_argument('param_file', type=str, help=h)
    
    h = 'the minimum k value to include in the fit'
    parser.add_argument('--kmin', type=float, default=0.01, help=h)
    
    h = 'the maximum k value to include in the fit'
    parser.add_argument('--kmax', type=float, default=0.6, help=h)

    h = 'the name of the output file to save the results to'
    parser.add_argument('-o', '--output', type=str, required=True, help=h)
    
    h = 'whether to append results to the existing data frame'
    parser.add_argument('--append', action='store_true', default=False, help=h)
    
    # model caching
    if cache_model:
        add_model_cache(parser)
    
    # add the samples
    for i, (dim, vals) in enumerate(zip(dims, coords)):
        h = 'the #%d bin coordinate values' %i
        parser.add_argument('--%s' %dim, nargs='+', choices=['all']+vals, help=h, default=['all'])
        
    if not return_parser:
        return parser.parse_args()
    else:
        return parser
    
    
def parse_global_mcmc(desc):
    """
    Return the parsed arguments for a global mcmc fit
    """
    parser = argparse.ArgumentParser(description=desc)
    
    h = 'the name of the mcmc parameter file to load'
    parser.add_argument('param_file', type=str, help=h)
    
    h = 'the minimum k value to include in the fit'
    parser.add_argument('--kmin', type=float, default=0.01, help=h)
    
    h = 'the maximum k value to include in the fit'
    parser.add_argument('--kmax', type=float, default=0.6, help=h)

    h = 'the name of the output file to save the results to'
    parser.add_argument('-o', '--output', type=str, required=True, help=h)
        
    # model caching
    add_model_cache(parser)
            
    return parser.parse_args()