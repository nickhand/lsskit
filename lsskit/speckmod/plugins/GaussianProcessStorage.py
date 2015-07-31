"""
    GaussianProcessStorage.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugin to fit a Gaussian Process to the best-fit parameters or functions
"""

from lsskit import numpy as np
from lsskit.speckmod.plugins import ModelResultsStorage
from sklearn.gaussian_process import GaussianProcess
import pandas as pd
import cPickle

def list_str(value):
    return value.split()

def list_float(value):
    return map(float, value.split())

class BestfitFunctionGPStorage(ModelResultsStorage):
    """
    Fit and save a Gaussian Process to the best-fit functions
    """
    name = "BestfitFunctionGPStorage"
    plugin_type = 'GP'
    
    @classmethod
    def register(cls):
        
        args = cls.name+":input:index_cols:output"
        options = "[:-regr=quadratic][:-random_start=1][:-thetaU= 1 1 1]"
        h = cls.add_parser(cls.name, usage=args+options)
        
        # arguments
        h.add_argument("input", type=str, help="the name of the input file to read")
        h.add_argument("index_cols", type=list_str, help="the names of index columns in input file")
        h.add_argument("output", type=str, help="the output name")
        
        # options
        choices = ['constant', 'linear', 'quadratic']
        h.add_argument('-regr', choices=choices, default='quadratic', 
                        help='the regression type to use')
        h.add_argument('-random_start', type=int, default=1, 
                        help='the random start to use')
        h.add_argument('-thetaU', type=float, default=1., 
                        help='the upper limits on the theta parameters to assume')
        
        h.set_defaults(klass=cls)
        
    def __open__(self):
        return None
        
    def __finalize__(self): 
        pass
        
    def write(self):
        
        # load the input data and set the index
        data = pd.read_pickle(self.input)
        data = data.set_index(self.index_cols)
        
        # get the kwargs for the fit
        kwargs = {}
        kwargs['corr'] = 'squared_exponential'
        kwargs['theta0'] = [0.1]*len(self.index_cols)
        kwargs['thetaL'] = [1e-4]*len(self.index_cols)
        kwargs['thetaU'] = [self.thetaU]*len(self.index_cols)
        kwargs['regr'] = self.regr
        kwargs['random_start'] = self.random_start
                    
        # compute the GP
        X = np.asarray(list(data.index.get_values()))
        y = data['mean'].values
        dy = data['error'].values
        kwargs['nugget'] = (dy/y)**2
        gp = GaussianProcess(**kwargs)

        # and fit
        gp.fit(X, y)
    
        # now save
        cPickle.dump(gp, open(self.output, 'w'))
        
#------------------------------------------------------------------------------
class BestfitParamGPStorage(ModelResultsStorage):
    """
    Fit and save a Gaussian Process to the best-fit parameters
    """
    name = "BestfitParamGPStorage"
    plugin_type = 'GP'
    
    @classmethod
    def register(cls):
        
        args = cls.name+":input:index_cols:param_names:output"
        options = "[:-regr=quadratic][:-random_start=1][:-thetaU= 1 1 1]"
        h = cls.add_parser(cls.name, usage=args+options)
        
        # arguments
        h.add_argument("input", type=str, help="the name of the input file to read")
        h.add_argument("param_names", type=list_str, help="the names of the parameter to fit")
        h.add_argument("index_cols", type=list_str, help="the names of index columns in input file")
        h.add_argument("output", type=str, help="the output name")
        
        # options
        choices = ['constant', 'linear', 'quadratic']
        h.add_argument('-regr', choices=choices, default='quadratic', 
                        help='the regression type to use')
        h.add_argument('-random_start', type=int, default=1, 
                        help='the random start to use')
        h.add_argument('-thetaU', type=float, default=1., 
                        help='the upper limits on the theta parameters to assume')
        
        h.set_defaults(klass=cls)
        
    def __open__(self):
        return None
        
    def __finalize__(self): 
        pass
        
    def write(self):
        
        # load the input data and set the index
        data = pd.read_pickle(self.input)
        data = data.set_index(self.index_cols)
        
        # get the kwargs for the fit
        kwargs = {}
        kwargs['corr'] = 'squared_exponential'
        kwargs['theta0'] = 0.1
        kwargs['thetaL'] = 1e-4
        kwargs['thetaU'] = self.thetaU
        kwargs['regr'] = self.regr
        kwargs['random_start'] = self.random_start
                    
        # compute the GP
        X = np.asarray(list(data.index.get_values()))
        
        toret = {}
        for name in self.param_names:
            
            y = data[name].values
            dy = data[name+'_err'].values
            kwargs['nugget'] = (dy/y)**2

            # make the GP and fit
            gp = GaussianProcess(**kwargs)
            gp.fit(X, y)
            toret[name] = gp
    
        # now save
        cPickle.dump(toret, open(self.output, 'w'))

        
        