"""
    tools.py
    lsskit.data

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : tools for dealing with the power spectra data
"""
from lsskit.data import PowerSpectraLoader
import argparse
from .. import numpy as np

def compute_pkmu_covariance(data, kbins=None, mubins=None, kmin=-np.inf, kmax=np.inf):
    """
    Compute covariance matrix of P(k,mu) measurements
    
    Parameters
    ----------
    data : SpectraSet
        a set of PkmuResult objects to compute the covariance from
    kbins : int or array_like, optional
        re-index the measurements using these k-bins
    mubins : int or array_like, optional
        re-index the measurements using these mu-bins
    kmin : float or array_like
        the minimum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    kmax : float or array_like
        the maximum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    
    Returns
    -------
    sizes : list
        list of the sizes of each submatrix a given mu bin
    ks : array_like
        the concatenated list of k values for each submatrix in the covariance matrix
    mus : array_like
        the concatenated list of mus values for each submatrix in the covariance matrix
    covar : array_like
        the covariance matrix
    """
    power, ks, mus, sizes = [], [], [], []
    for i, p in enumerate(data):
        p = p.values
        
        # get the kmin/kmax as arrays
        kmin_ = np.empty(p.Nmu)
        kmax_ = np.empty(p.Nmu)
        kmin_[:] = kmin
        kmax_[:] = kmax
        
        # reindex k or mu bins
        if kbins is not None:
            p = p.reindex_k(kbins, weights='modes')
        if mubins is not None:
            p = p.reindex_mu(mubins, weights='modes')
            
        # get the valid entries and flatten so mus are stacked in order
        x, y, z = [], [], []
        for imu in range(p.Nmu):
            valid = get_valid_data(p.Pk(imu), kmin=kmin_[imu], kmax=kmax_[imu])
            x += list(valid['k'])
            y += list(valid['mu'])
            z += list(valid['power'])
            if i == 0:
                sizes.append(len(valid['k']))
            
        ks.append(x)
        mus.append(y)
        power.append(z)

    ks = np.asarray(ks)
    mus = np.asarray(mus)
    power = np.asarray(power)
    
    C = np.cov(power, rowvar=False)
    return sizes, ks.mean(axis=0), mus.mean(axis=0), C
    

def get_valid_data(data, kmin=None, kmax=None):
    """
    Return the valid data. First, any NaN entries are removed
    and if ``kmin`` or ``kmax`` are not ``None``, then the 
    ``k`` column in ``data`` is used to trim the valid range.
    
    Parameters
    ----------
    data : PkmuResult or PkResult
        The power data instance holding the `k`, `power`, and
        optionally, the `error` data columns
    kmin : float, optional
        minimum wavenumber to trim by (inclusively), in h/Mpc
    kmax : float, optional
        maximum wavenumber to trim by (inclusively), in h/Mpc
    
    Returns
    -------
    toret : dict
        dictionary holding the trimmed data arrays, with keys
        ``k``, ``power``, and optionally, ``error`` and ``mu``.
    """
    if hasattr(data, 'dtype') and hasattr(data.dtype, 'names'):
        columns = data.dtype.names
        shape = np.shape(data)
    else:
        columns = data.columns
        data = data.data
        shape = data.shape
       
    valid = np.ones(shape, dtype=bool)
    for col in columns:
        valid &= ~np.isnan(data[col])    
    if kmin is not None:
        valid &= (data['k'] >= kmin)
    if kmax is not None:
        valid &= (data['k'] <= kmax)
    
    # collapse across all dimensions  
    if valid.ndim > 1:
        valid = np.all(valid, axis=-1)
    shape = list(data[columns[0]].shape)
    shape[0] = valid.sum()
    
    # make the output
    dtype = [(name, 'f8') for name in columns]
    toret = np.empty(shape, dtype=dtype)
    for col in columns:
        toret[col] = data[col][valid,...]
    return toret

def get_Pshot(power):
    """
    Return the shot noise from a power spectrum instance
    """
    if hasattr(power, 'volume'):
        return power.volume / power.N1
    elif hasattr(power, 'box_size'):
        return power.box_size**3 / power.N1
    elif all(hasattr(power, x) for x in ['Lx', 'Ly', 'Lz']):
        return power.Lx*power.Ly*power.Lz / power.N1
    else:
        raise ValueError("cannot compute shot noise")

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
        if isinstance(values, basestring):
            values = [values]
        for value in values:
            key, val = value.split('=')
            val = map(eval, val.split(','))
            getattr(namespace, self.dest)[key.strip()] = val
            
class ReindexDict(argparse.Action):
    """
    Read reindex keys into a dictionary, optionally using numpy
    
    "k = numpy.arange(0.01, 0.5 0.01)" 
    """
    def __call__(self, parser, namespace, values, option_string=None):
        from lsskit import numpy    
        if getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, {})
        if isinstance(values, basestring):
            values = [values]
        for value in values:
            key, val = value.split('=')
            val = eval(val)
            if not (len(val)-1): val = val[0]
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
            
    