"""
    BestfitFunctionStorage.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugin to save the bestfit function values as a pandas dataframe
"""
from lsskit import numpy as np
from lsskit.speckmod.plugins import ModelResultsStorage
import pandas as pd
import itertools

#-------------------------------------------------------------------------------
def one_sigma(trace):
    percentiles = [50., 15.86555, 84.13445]
    vals = np.percentile(trace, percentiles)
    return [vals[2] - vals[0], vals[0] - vals[1]]

#-------------------------------------------------------------------------------
def get_one_sigma_errs(ks, params, model, **kwargs):
    """
    Return the function mean and 1-sigma error at each `k` value
    """         
    data = {name : params[name] for name in params.columns}
    data.update(kwargs)
    def function(data, k=None):
        values = model(k, **data)
        errs = one_sigma(values)
        return 0.5*(errs[0] + errs[1])
        
    return np.array([function(data, k=k) for k in ks])

#-------------------------------------------------------------------------------
def list_str(value):
    return value.split()
    
class BestfitFunctionStorage(ModelResultsStorage):
    """
    Write out the best-fit function values, as computed from an MCMC run, 
    to a pandas DataFrame
    """
    name = "BestfitFunctionStorage"
    plugin_type = 'output'
    
    @classmethod
    def register(cls):
        
        usage = cls.name+":path:index_cols"
        h = cls.add_parser(cls.name, usage=usage)
        h.add_argument("path", type=str, help="the output name")
        h.add_argument("index_cols", type=list_str, 
                help='the names of the fields to store as columns for later indexing')
        h.set_defaults(klass=cls)
        
    def __open__(self):
        try:
            return self._output
        except AttributeError:
            self._output = pd.DataFrame()
            return self._output
        
    def __finalize__(self):
        self._output.to_pickle(self.path)
        
    def write(self, key, result):
        from fitit import EmceeResults
        if not isinstance(result, EmceeResults):
            raise TypeError("`result` object in BestfitFunctionStorage.write must be a fitit.EmceeResults class")
        
        with self.open() as output:
                        
            # columns
            names = result.free_param_names
            columns = list(itertools.chain(*[(k, k+"_err") for k in names]))
            
            # data values
            vals = [(result[k].value, result[k].stderr) for k in names]
            data = tuple(itertools.chain(*vals))
            ks = result.indep_vars['k']

            # the function mean
            mu = result.best_fit
        
            # get the 1sigma errors
            traces = [result[k].flat_trace for k in names]
            other_names = [name for name in result.param_names if name not in names]
            traces += [np.repeat(result[name].value, len(traces[0])) for name in other_names]
            
            params = np.vstack(traces).T
            params = pd.DataFrame(params, columns=names+other_names)
            extra_kwargs = getattr(result.model, 'extra_kwargs', {})
            mean_errs = get_one_sigma_errs(ks, params, result.model.func, **extra_kwargs)
            
            # append to the output frame
            d = {k:np.repeat(key[k], len(ks)) for k in self.index_cols}
            d['mean'] = mu
            d['error'] = mean_errs
            d['k'] = ks
            self._output = output.append(pd.DataFrame(d))
            


        
        