"""
    BestfitParamStorage.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugin to save the bestfit parameters as a pandas dataframe
"""

from lsskit.speckmod.plugins import ModelResultsStorage
import pandas as pd
import itertools

class BestfitParamStorage(ModelResultsStorage):
    """
    Write out the best-fit parameters and errors to a pandas DataFrame
    """
    name = "BestfitParamStorage"
    plugin_type = 'output'
    
    @classmethod
    def register(cls):
        
        usage = cls.name+":path"
        h = cls.add_parser(cls.name, usage=usage)
        h.add_argument("path", type=str, help="the output name")
        h.set_defaults(klass=cls)
        
    def __open__(self):
        try:
            return self._output
        except AttributeError:
            self._output = pd.DataFrame(index=self.index)
            return self._output
        
    def __finalize__(self):
        self._output.to_csv(self.path, sep=" ", float_format="%.4e")
        
    def write(self, index, key, result):
        
        if not hasattr(self, 'index'): self.index = index
        with self.open() as output:
            
            # transform the key
            key = tuple(key[k] for k in self.index.names)
        
            # columns
            names = result.param_names
            columns = list(itertools.chain(*[(k, k+"_err") for k in names]))
            
            # data values
            vals = [(result[k].value, result[k].stderr) for k in names]
            data = tuple(itertools.chain(*vals))
            
            # save
            for col, v in zip(columns, data):
                output.loc[key, col] = v
        

        
        