"""
    BestfitParamStorage.py
    lsskit.speckmod.plugins
    
    __author__ : Nick Hand
    __desc__ : plugin to save the bestfit parameters as a pandas dataframe
"""

from lsskit.speckmod.plugins import ModelResultsStorage
import pandas as pd
import itertools

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
        
        usage = cls.name+":path:index_cols"
        h = cls.add_parser(cls.name, usage=usage)
        h.add_argument("path", type=str, help="the output name")
        h.add_argument("index_cols", type=list_str, help="the names of the columns to index by")
        h.set_defaults(klass=cls)
        
    def __open__(self):
        try:
            return self._output
        except AttributeError:
            self._output = pd.DataFrame()
            return self._output
        
    def __finalize__(self):
        
        # reindex properly b/c pandas is stupid
        index = [v for v in self._output.index.values if len(v) == len(self.index_cols)]
        mi = pd.MultiIndex.from_tuples(index, names=self.index_cols)
        self._output = pd.DataFrame(self._output.loc[index], index=mi)
        
        # and save
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
            
            # make a new dataframe and save
            df = pd.DataFrame(dict(zip(columns, data)), index=pd.Index([key]))
            self._output = output.append(df)

        

        
        