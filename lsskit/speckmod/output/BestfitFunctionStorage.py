from lsskit import numpy as np
import pandas as pd
import itertools
import os
import contextlib 

def one_sigma(trace):
    percentiles = [50., 15.86555, 84.13445]
    vals = np.percentile(trace, percentiles)
    return [vals[2] - vals[0], vals[0] - vals[1]]

def get_one_sigma_errs(indep_vars, params, model, **kwargs):
    """
    Return the function mean and 1-sigma error at each `k` value
    """         
    data = {name : params[name] for name in params.columns}
    data.update(kwargs)
    def function(data, **indep_vars):
        data.update(indep_vars)
        values = model(**data)
        errs = one_sigma(values)
        return 0.5*(errs[0] + errs[1])
    
    names = indep_vars.keys()    
    indep_vars = np.vstack([v for v in indep_vars.values()]).T
    return np.array([function(data, **dict(zip(names, row))) for row in indep_vars])

    
class BestfitFunctionStorage(object):
    """
    Write out the best-fit function values, as computed from an MCMC run, 
    to a pandas DataFrame
    """
    def __init__(self, path, append=False):
        
        self.path   = path
        self.append = append
        if self.append and not os.path.exists(self.path):
            raise ValueError("cannot append to non-existing output file `%s`" %self.path)
        
    @contextlib.contextmanager
    def open(self, index_cols):
        output = self.__open__()
        try:
            yield output
        finally:
            self.__finalize__(index_cols)
            
    def __open__(self):
        try:
            return self._output
        except AttributeError:
            if os.path.exists(self.path) and self.append:
                self._output = pd.read_pickle(self.path)
            else:
                self._output = pd.DataFrame()
            return self._output
        
    def __finalize__(self, index_cols):
        if self.append:
            self._output = self._output.drop_duplicates(subset=index_cols, take_last=True)
        self._output.index = range(len(self._output)) # reset index back to 0 - N
        self._output.to_pickle(self.path)
        
    def write(self, key, result, model, eval_kws, **extra):
        
        if 'k' not in key:
            raise ValueError("to write the best-fit function must have `k` in the key")
            
        index_cols = key.keys()
        with self.open(index_cols) as output:

            Nk = len(key['k'])
            
            # the function mean
            kws = eval_kws.copy()
            params = dict(zip(result.free_names, result.values()))
            kws.update(params)
            mu = model.eval(**kws)
        
            # get the 1sigma errors
            traces = [result[k].flat_trace for k in result.free_names]
            other_names = [name for name in model.param_names if name not in result.free_names]
            traces += [np.repeat(result[name].median, len(traces[0])) for name in other_names]
            
            params = np.vstack(traces).T
            params = pd.DataFrame(params, columns=result.free_names+other_names)
            indep_vars = {k:eval_kws[k] for k in model.independent_vars}
            mean_errs = get_one_sigma_errs(indep_vars, params, model.eval, **eval_kws)
            
            # append to the output frame
            d = {'mean':mu, 'error':mean_errs}
            for k in key:
                if np.isscalar(key[k]):
                    d[k] = np.repeat(key[k], Nk)
                else:
                    d[k] = key[k]
            for k in extra:
                d[k] = np.repeat(extra[k], Nk)
            self._output = output.append(pd.DataFrame(d))
            


        
        