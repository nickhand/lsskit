import argparse

def parse_mcmc_gp(desc):
    """
    Return the parsed arguments for the GP fit which uses `emcee` 
    to perform the hyperparameter fit
    """
    kws = {'formatter_class':argparse.ArgumentDefaultsHelpFormatter}
    parser = argparse.ArgumentParser(description=desc, **kws)
    
    h = 'the path to the data to load'
    parser.add_argument('data_path', type=str, help=h)
    
    h = 'the name of the data column to fit'
    parser.add_argument('column', type=str, help=h)
    
    h = 'the output path to save the best-fit hyperparameters'
    parser.add_argument('-o', '--output', type=str, required=True, help=h)
    
    h = 'the number of iterations to run'
    parser.add_argument('-i', '--iterations', type=int, default=100, help=h)
    
    h = 'the number of burnin steps to run; default is same as iterations'
    parser.add_argument('-b', '--burnin', type=int, help=h)
    
    h = 'the number of walkers to run with'
    parser.add_argument('-n', '--nwalkers', type=int, default=36, help=h)
    
    h = 'whether we want to vary the amplitude of the GP kernel'
    parser.add_argument('--vary-amplitude', action='store_true', help=h)
        
    return parser.parse_args()
    
    
def parse_ml_gp(desc):
    """
    Return the parsed arguments for the GP fit which uses 
    `scipy.optimize` to find the maximum likelihood solution
    """
    kws = {'formatter_class':argparse.ArgumentDefaultsHelpFormatter}
    parser = argparse.ArgumentParser(description=desc, **kws)
    
    h = 'the path to the data to load'
    parser.add_argument('data_path', type=str, help=h)
    
    h = 'the name of the data column to fit'
    parser.add_argument('column', type=str, help=h)
    
    h = 'the output path to save the best-fit hyperparameters'
    parser.add_argument('-o', '--output', type=str, required=True, help=h)
    
    h = 'the initial (log) theta values to start from'
    parser.add_argument('--log_theta0', nargs='*', type=float, help=h)
    
    h = 'the optimization method to use'
    parser.add_argument('--method', type=str, default='NEWTON-CG', help=h)
    
    h = 'the number of random starts to use'
    parser.add_argument('--random_start', type=int, default=1, help=h)
    
    h = 'whether we want to vary the amplitude of the GP kernel'
    parser.add_argument('--vary-amplitude', action='store_true', help=h)
        
    return parser.parse_args()