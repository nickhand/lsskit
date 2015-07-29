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
        h.add_argument("index_cols", type=list_str, help="the names of the columns to index by")
        h.set_defaults(klass=cls)
        
    def __open__(self):
        try:
            return self._output
        except AttributeError:
            columns = ['mean', 'error']
            self._output = pd.DataFrame(columns=columns)
            return self._output
        
    def __finalize__(self):

        # reindex properly b/c pandas is stupid
        index = [v for v in self._output.index.values if len(v) == len(self.index_cols)+1]
        mi = pd.MultiIndex.from_tuples(index, names=self.index_cols+['k'])
        self._output = pd.DataFrame(self._output.loc[index], index=mi)

        # save
        self._output.to_csv(self.path, sep=" ", float_format="%.4e")
        
    def write(self, key, result):
        
        with self.open() as output:
            
            # transform the key
            key = tuple(key[k] for k in self.index_cols)
        
            # columns
            names = result.param_names
            columns = list(itertools.chain(*[(k, k+"_err") for k in names]))
            
            # data values
            vals = [(result[k].value, result[k].stderr) for k in names]
            data = tuple(itertools.chain(*vals))
            ks = result.indep_vars['k']
            
            # new index with `k` as a column
            new_index = [key+(ki,) for ki in ks]
            index_names = self.index_cols+['k']
            
            # the function mean
            mu = result.best_fit
        
            # get the 1sigma errors
            params = np.vstack([result[k].flat_trace for k in names]).T
            params = pd.DataFrame(params, columns=names)
            extra_kwargs = getattr(result.model, 'extra_kwargs', {})
            mean_errs = get_one_sigma_errs(ks, params, result.model.func, **extra_kwargs)
            
            # append to the output frame
            d = {'mean':mu, 'error':mean_errs}
            df = pd.DataFrame(d, index=pd.MultiIndex.from_tuples(new_index, names=index_names))
            self._output = output.append(df)
            


        
        