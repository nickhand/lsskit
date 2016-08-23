import argparse

def parse_binned_mcmc(desc, dims, coords, return_parser=False):
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
    
    # add the samples
    for i, (dim, vals) in enumerate(zip(dims, coords)):
        h = 'the #%d bin coordinate values' %i
        parser.add_argument('--%s' %dim, nargs='+', choices=['all']+vals, help=h, default=['all'])
        
    if not return_parser:
        return parser.parse_args()
    else:
        return parser
    
    
def parse_global_mcmc(desc, return_parser=False):
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
         
    if not return_parser:
        return parser.parse_args()
    else:
        return parser   