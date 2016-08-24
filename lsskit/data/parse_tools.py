"""
    tools.py
    lsskit.data

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : tools for dealing with the power spectra data
"""
from lsskit.data import PowerSpectraLoader
import argparse

def parse_options(options):
    """
    Given a list of strings specifying command-line options, parse and
    return a dictionary with the key/value pairs
    
    Notes
    -----
    All options must have values, i.e., no flag-like options can be parsed
    
    Parameters
    ----------
    options : list of str
        a list of strings specifying the command line options, i.e., 
        in the format ``--option=val`` or ``--option val``
    
    Returns
    -------
    kwargs : dict
        a dictionary holding the option key/value pairs
    """
    kwargs = {}
    for w in options:
        if '=' in w:
            fields = w.split('=')
        else:
            fields = w.split()
        if len(fields) != 2:
            raise ValueError("all  %s" %w)
        
        k, v = fields
        if '--' in k:
            k = k.split('--')[-1].strip()
        else:
            k = k.split('-')[-1].strip()
        kwargs[k] = eval(v.strip())
    return kwargs

class PowerSpectraParser(object):
    """
    A class to parse a ``lsskit.data`` plugin from the command-line
    and return the instance via the ``data`` method
    """
    def __init__(self, string):
        words = string.split(":")
        name = words.pop(0).strip()
        root_dir = words.pop(0).strip()
        kwargs = parse_options(words)
        self._data = PowerSpectraLoader.get(name.strip(), root_dir.strip(), **kwargs)
    
    @classmethod
    def data(cls, string):
        return cls(string)._data
        
    @classmethod
    def format_help(cls):
        h = "the name of the power spectra data to load:\n"
        usage = "\tusage: name:root_dir[--plugin_path=None][**kwargs]"
        return h+usage
    
class PowerSpectraCallable(object):
    """
    A class to parse a method of a ``lsskit.data`` plugin
    and return the name and any optional keyword values
    """
    def __init__(self, string):
        words = string.split(":")
        name = words.pop(0).strip()
        kwargs = parse_options(words)
        self._data = {'name':name, 'kwargs':kwargs}
        
    @classmethod
    def data(cls, string):
        return cls(string)._data
        
    @classmethod
    def format_help(cls):
        h = "function name specifying to call to return power spectra data:\n"
        usage = "\tusage: name:[**kwargs]"
        return h+usage
        
class StoreDataKeys(argparse.Action):
    """
    Read data keys from the command line, as a comma-seperated list:
    
    "key = val1, val2, val3" 
    """
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        if isinstance(values, str):
            values = [values]
        for value in values:
            key, val = value.split('=')
            val = map(eval, val.split(','))
            getattr(namespace, self.dest)[key.strip()] = val
            
class AliasAction(argparse.Action):
    """
    Parse aliases into a dictionay, assuming the format:
    
    --option "key1:alias1, key2:alias2"
    """
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        key, val = values.split('=')
        pairs = val.split(',')
        val = dict([tuple(eval(y.strip()) for y in x.split(':')) for x in pairs])
        getattr(namespace, self.dest)[key.strip()] = val
            
    