"""
    utils.py
    lsskit.specksis
    
    __author__ : Nick Hand
    __desc__ : internal utilities for helping with analysis of power spectra
"""
import os
import itertools
from .. import numpy as np
          
def ndindex(dims, coords):
    """
    Generator to iterate over the all dimensions using the specified coordinate
    values. The function takes the product of each coordinate axis and
    returns a dictionary with ``dims`` as the keys and the corresponding
    coordinates as values
    
    Parameters
    ----------
    dims : list of str
        the names of the dimensions; the keys in the yielded dictionary
    coordinates : list, xray.core.coordinates.DataArrayCoordinates
        list of coordinates for each dimension. `itertools.product` of
        each coordinate axis is yielded.
    """
    for idx, args in ndenumerate(dims, coords):
        yield args
        
def ndenumerate(dims, coords):
    """
    Generator to enumerate all dimensions using the specified coordinate
    values. The function takes the product of each coordinate axis and
    returns a dictionary with ``dims`` as the keys and the corresponding
    coordinates as values
    
    Parameters
    ----------
    dims : list of str
        the names of the dimensions; the keys in the yielded dictionary
    coordinates : list, xray.core.coordinates.DataArrayCoordinates
        list of coordinates for each dimension. `itertools.product` of
        each coordinate axis is yielded.
    """
    if not is_dict_like(coords):
        if len(dims) != len(coords):
            raise ValueError("shape mismatch between supplied `dims` and `coords`")
        coords = dict(zip(dims, coords))
    dims = list(dims)
    if not len(dims):
        raise ValueError("cannot iterate if no dimension with size > 1 is provided")
    coords = [coords[d] for d in dims]
       
    # make the array of dicts 
    shape, ndims = map(len, coords), len(dims)
    values = itertools.product(*[coords[i] for i in range(ndims)])
    allargs = np.array([dict(zip(dims, v)) for v in values]).reshape(*shape)
    
    for idx in np.ndindex(*shape):
        args = allargs[idx]
        yield idx, args

def is_dict_like(value):
    return hasattr(value, '__getitem__') and hasattr(value, 'keys')
    
def enum_files(result_dir, basename, dims, coords):
    """
    A generator to enumerate over all files in `result_dir`,
    starting with `basename`. The string `basename` is formatted
    using `string.format`. The keys used to format are those
    in `dims` and the values are all combinations of the 
    coordinate values in `coords`.
    
    Notes
    -----
    This yields the index and the filename, where the index is 
    the N-dimensional index as returned by numpy.ndindex. The
    shape of the N-d index is the length of each component of 
    `coords`
    """
    # shape of data
    shape = map(len, coords)
    
    # compute the list of dicts for string formatting
    ndims = len(dims)
    values = list(itertools.product(*[coords[i] for i in range(ndims)]))
    allargs = np.array([dict(zip(dims, v)) for v in values]).reshape(*shape)
    
    # get abs paths to directories
    result_dir = os.path.abspath(result_dir)

    # try to find all files
    for idx, args in ndenumerate(dims, coords):
        try:            
            # see if path exists
            f = os.path.join(result_dir, basename).format(**args)
            if not os.path.exists(f): raise
        
            # yield index and filename
            yield idx, f
        
        except:
            message = 'no file found for `%s`\n in directory `%s`' %(basename, result_dir)
            raise IOError(message)

def add_errors(power, power_x1=None, power_x2=None):
    """
    Add the error on power spectrum measurement as the column `error`
    to the input ``power`` object.

    Notes
    -----
    This modifies the power spectrum object in place
    
    Parameters
    ----------
    power : nbodykit.PkmuResult, nbodykit.PkResult
        the power spectrum object. must have `modes` data column
    power_x1, power_x2 : nbodykit.PkmuResult, nbodykit.PkResult, optional
        If ``power`` stores a cross-power measurement, the auto power
        measurements are needed to compute the error
    """
    if 'modes' not in power:
        raise ValueError("need `modes` data column to compute errors")
        
    # auto power calculation
    # error = sqrt(2/N_modes) * Pxx
    if power_x1 is None and power_x2 is None:
        with np.errstate(invalid='ignore'):
            err = (2./power['modes'])**0.5 * power['power']
    # cross power calculation
    # error = sqrt(1/N_modes) * (Pxy + sqrt(Pxx*Pyy))
    else:
        
        with np.errstate(invalid='ignore'):
            err = (1./power['modes'])**0.5 * power['power']
            err += (1./power['modes'])**0.5 * (power_x1['power']*power_x2['power'])**0.5 
            
    power.add_column('error', err)  
    
def load_data_from_file(filename, dims, shape):
    """
    Load a pickled dictionary of linear biases and return
    a `xray.DataArray`
    
    Parameters
    ----------
    filename : str
        the name of the file holding the pickled data
    dims : list of str
        the list of strings corresponding to the names of
        the dimensions of the dictionary keys
    shape : list of int
        the shape of the data values, corresponding to the
        shape of dim 0, dim 1, etc
    """
    import pickle
    import xray
    
    # load the data
    biases = pickle.load(open(filename))
        
    # sort keys and values by the keys
    keys = biases.keys()
    b1 = biases.values()
    sorted_lists = sorted(zip(keys, b1), key=lambda x: x[0])
    keys, b1 = [[x[i] for x in sorted_lists] for i in range(2)]

    # make the coords and return a DataArray
    coords = [keys]
    if len(dims) > 1:
        coords = zip(*keys)
        coords = [np.unique(x) for x in coords]
        b1 = np.array(b1).reshape(shape)
    return xray.DataArray(b1, coords, dims)

