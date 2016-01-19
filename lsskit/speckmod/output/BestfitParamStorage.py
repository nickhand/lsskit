"""
    BestfitParamStorage.py
    lsskit.speckmod.output
    
    __author__ : Nick Hand
    __desc__ : plugin to save the bestfit parameters as a pandas dataframe
"""
from lsskit import numpy as np
import pandas as pd
import itertools
import os
import contextlib

class BestfitParamStorage(object):
    """
    Store best-fit parameters and errors to a `pandas.DataFrame`
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
        
    def write(self, key, result, **extra):
        """
        Append a row to the output DataFrame, given 
        the bin `key` and mcmc `result`
        """
        index_cols = key.keys()
        with self.open(index_cols) as output:
                    
            # columns
            names = result.free_names
            columns = list(itertools.chain(*[(k, k+"_err") for k in names]))
            
            # data values
            vals = [(result[k].median, result[k].stderr) for k in names]
            data = tuple(itertools.chain(*vals))
            
            # make a new dataframe and save
            d = dict(zip(columns, data))
            
            # add the extra info
            extra.update(key)
            d.update(extra)

            self._output = output.append(pd.DataFrame(d, index=[len(output)]))


        

        
        
