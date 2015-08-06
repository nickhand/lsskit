"""
    InterpolationStorage.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugin to store interpolation schemes of the best-fit parameters or functions
"""
from lsskit import numpy as np
from lsskit.speckmod.plugins import ModelResultsStorage

from sklearn.gaussian_process import GaussianProcess
from scipy.interpolate import UnivariateSpline as spline
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
    plugin_type = 'gp_interp'
    
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
    plugin_type = 'gp_interp'
    
    @classmethod
    def register(cls):
        
        args = cls.name+":input:param_names:index_cols:output"
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
        kwargs['theta0'] = [0.1]*len(self.index_cols)
        kwargs['thetaL'] = [1e-4]*len(self.index_cols)
        kwargs['thetaU'] = [self.thetaU]*len(self.index_cols)
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
        
#------------------------------------------------------------------------------        
class BestfitParamSplineStorage(ModelResultsStorage):
    """
    Fit and save a series of splines to the best-fit functions
    """
    name = "BestfitParamSplineStorage"
    plugin_type = 'spline_interp'
    
    @classmethod
    def register(cls):
        
        args = cls.name+":input:param_names:index_cols:output"
        options = "[:-use_errors]"
        h = cls.add_parser(cls.name, usage=args+options)
        
        # arguments
        h.add_argument("input", type=str, help="the name of the input file to read")
        h.add_argument("param_names", type=list_str, help="the names of the parameters to fit")
        h.add_argument("index_cols", type=list_str, help="the names of index columns in input file")
        h.add_argument("output", type=str, help="the output name")
        
        # options
        h.add_argument('-use_errors', action='store_true', default=False, 
                        help='the regression type to use')
        
        h.set_defaults(klass=cls)
        
    def __open__(self):
        return None
        
    def __finalize__(self): 
        pass
        
    def write(self):
        
        if len(self.index_cols) != 2:
            raise ValueError("spline storage only supports exactly two index columns")
        table = {}
        
        # load the input data and set the index
        data = pd.read_pickle(self.input)
        data = data.set_index(self.index_cols)
        
        for key in data.index.levels[0]:
            table[key] = {}
            
            # loop over each parameter
            for name in self.param_names:
            
                d = data.xs(key)
                
                # get the data to be interpolated, making sure to remove nulls
                y = d[name]
                null_inds = y.notnull() 
                y = np.array(y[null_inds])
        
                # X values are only b1 not sigma8 too
                X = np.array(d.index)           
  
                # check for error columns
                kwargs = {'k' : 2}
                if self.use_errors and name+'_err' in data:
                    dy = d[name+'_err'][null_inds]
                    kwargs['w'] = 1.0/dy

                # set the spline
                table[key][name] = spline(X, y, **kwargs)

        # now save
        cPickle.dump(table, open(self.output, 'w'))
        

        
        