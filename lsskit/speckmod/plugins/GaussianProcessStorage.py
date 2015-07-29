"""
    GaussianProcessStorage.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugin to fit a Gaussian Process to the best-fit parameters or functions
"""

from lsskit.speckmod.plugins import ModelResultsStorage
from sklearn.gaussian_process import GaussianProcess
import pandas as pd
import cPickle

def list_str(value):
    return value.split()

def list_float(value):
    return map(float, value.split())

class GaussianProcessStorage(ModelResultsStorage):
    """
    Fit and save a Gaussian Process to the best-fit parameters or functions
    """
    name = "GaussianProcessStorage"
    plugin_type = 'GP'
    
    @classmethod
    def register(cls):
        
        args = cls.name+":kind:input:index_cols:output"
        options = "[:-regr=quadratic][:-random_start=1][:-thetaU= 1 1 1]"
        h = cls.add_parser(cls.name, usage=args+options)
        
        # arguments
        h.add_argument("kind", type=str, choices=['functions', 'parameters'], help="fit parameters or functions")
        h.add_argument("input", type=str, help="the name of the input file to read")
        h.add_argument("index_cols", type=list_str, help="the names of index columns in input file")
        h.add_argument("output", type=str, help="the output name")
        
        # options
        choices = ['constant', 'linear', 'quadratic']
        h.add_argument('-regr', choices=choices, default='quadratic', 
                        help='the regression type to use')
        h.add_argument('-random_start', type=int, default=1, 
                        help='the random start to use')
        h.add_argument('-thetaU', type=list_float, default=[1., 1., 1.], 
                        help='the upper limits on the theta parameters to assume')
        
        h.set_defaults(klass=cls)
        
    def __open__(self):
        return None
        
    def __finalize__(self): 
        pass
        
    def write(self, key, result):
        
        # load the input data
        data = pd.read_csv(self.input, delim_whitespace=True, index_col=self.index_cols)
        
        # get the kwargs for the fit
        kwargs = {}
        kwargs['corr'] = 'squared_exponential'
        kwargs['theta0'] = [0.1, 0.1, 0.1]
        kwargs['thetaL'] = [1e-4, 1e-4, 1e-4]
        kwargs['thetaU'] = self.thetaU
        kwargs['regr'] = self.regr
        kwargs['random_start'] = self.random_start
                    
        # compute the GP
        X = np.asarray(list(data.index.get_values()))
        y = data['mean']
        dy = df['error']
        kwargs['nugget'] = (dy/y)**2
        gp = GaussianProcess(**kwargs)

        # and fit
        gp.fit(X, y)
    
        # now save
        cPickle.dump(gp, open(self.output, 'w'))

        
        