"""
    BestfitParamStorage.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugin to save the bestfit parameters as a pandas dataframe
"""

from lsskit.speckmod.plugins import ModelResultsStorage
from lsskit import numpy as np
import pandas as pd
import itertools
import os

def list_str(value):
    return value.split()

class BestfitParamStorage(ModelResultsStorage):
    """
    Write out the best-fit parameters and errors to a pandas DataFrame
    """
    name = "BestfitParamStorage"
    plugin_type = 'output'
    
    @classmethod
    def register(cls):
        
        usage = cls.name+":path:index_cols[:-append]"
        h = cls.add_parser(cls.name, usage=usage)
        h.add_argument("path", type=str, help="the output name")
        h.add_argument("index_cols", type=list_str, 
                help='the names of the fields to store as columns for later indexing')
        h.add_argument("-append", action='store_true', default=False, 
                help='append the results to the output dataframe')
        h.set_defaults(klass=cls)
        
    def __open__(self):
        try:
            return self._output
        except AttributeError:
            if os.path.exists(self.path) and self.append:
                self._output = pd.read_pickle(self.path)
            else:
                self._output = pd.DataFrame()
            return self._output
        
    def __finalize__(self):
        if self.append:
            self._output = self._output.drop_duplicates(subset=self.index_cols, take_last=True)
        self._output.to_pickle(self.path)
        
    def write(self, key, result):
        
        with self.open() as output:
                    
            # columns
            names = result.param_names
            columns = list(itertools.chain(*[(k, k+"_err") for k in names]))
            
            # data values
            vals = [(result[k].value, result[k].stderr) for k in names]
            data = tuple(itertools.chain(*vals))
            
            # make a new dataframe and save
            d = dict(zip(columns, data))
            d.update({k:key[k] for k in self.index_cols})
            for k in d:
                d[k] = np.array(d[k], ndmin=1)
            self._output = output.append(pd.DataFrame(d, index=[len(self._output)]))


        

        
        