"""
    modeling.py
    
    __author__ : Nick Hand
    __desc__ : functions for helping with modeling of power spectra
"""
from .. import numpy as np

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
        ``k``, ``power``, and optionally, ``error``.
    """
    columns = ['k', 'power', 'error']
    
    valid = ~np.isnan(data['power'])
    if kmin is not None:
        valid &= (data['k'] >= kmin)
    if kmax is not None:
        valid &= (data['k'] <= kmax)
    
    toret = {}
    for col in columns:
        if col in data:
            toret[col] = data[col][valid]

    return toret

def make_param_table(param_names, dims, coords):
    """
    Return an empty ``pandas.DataFrame``, indexed by a ``MultiIndex``
    specified by ``dims`` and ``coords`` and with column names for 
    each parameter (and error) in ``param_names``.
    
    Parameters
    ----------
    param_names : list of str
        the names of the parameters, which will serve as the column names
    dims : list of str
        the names of the dimensions of the MultiIndex
    coordinates : list, xray.core.coordinates.DataArrayCoordinates
        list of coordinates for each dimension. `itertools.product` of
        each coordinate axis is used to make the MultiIndex
    
    Returns
    -------
    df : pandas.DataFrame
        an empty DataFrame to store the value and error on each parameter
    """
    import itertools
    import pandas as pd
    
    if hasattr(coords, 'values'):
        coords = [x.values for x in coords.values()]
        
    param_plus_errs = list(itertools.chain(*[(p, p+"_err") for p in param_names]))
    index = list(itertools.product(*[coords[i] for i in range(len(dims))]))
    index = pd.MultiIndex.from_tuples(index, names=dims)
    return pd.DataFrame(index=index, columns=param_plus_errs)