"""
    tools.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : analysis tools
"""
from .. import numpy as np

def flat_and_nonnull(arr):
    """
    Flatten the input array using `Fortran` format 
    (i.e., col #1 then col #2, etc), and remove 
    any NaNs along the way
    """
    flat = arr.ravel(order='F')
    return flat[np.isfinite(flat)]

def stack_multipoles(pole_set, ells=None):
    """
    Given a SpectraSet with an `ell` dimension, stack those individual
    P(k) results into a 2D array where the second dimension is the 
    original `ell` dimension. 
    
    Notes
    -----
    Additional dimensions in `pole_set` are preserved, as the higher
    order dimensions, where the first two are ('k', 'ell')
    
    Returns
    -------
    toret : np.ndarray
        a structured array holding the newly stacked data
    """
    # the dimension to loop over
    if ells is None: ells = pole_set['ell'].values
    dims = list(pole_set.dims)
    dims.remove('ell')

    def stack_one(poles):
        tostack = []
        for ell in ells:
            p = poles.sel(ell=int(ell)).values
            tostack.append(p.data.data)
        return np.vstack(tostack).T

    if len(dims):
        other_dim = dims[0]
        toret = []
        for i in pole_set[other_dim]:
            poles = pole_set.sel(**{other_dim:i})
            toret.append(stack_one(poles))
        toret = np.asarray(toret)
        return np.rollaxis(toret, 0, toret.ndim)
    else:
        return stack_one(pole_set)
             
def format_multipoles(this_pole, this_pkmu, ells):
    """
    Format the input spectra of multipoles, which includes multiple
    multipoles within one `PkResult`, such that the multipole number
    has its own dimension. Also add errors based on the specified
    `P(k,mu)` results
    """
    from scipy.special import legendre
    from nbodykit import pkresult

    tags, ells = zip(*ells)

    meta = {k:getattr(this_pole, k) for k in this_pole._metadata}
    meta['edges'] = this_pole.kedges
    
    # weight by modes
    modes = np.nan_to_num(this_pkmu['modes'].data)
    N_1d = modes.sum(axis=-1)
    weights = modes / N_1d[:,None]

    # avg mu
    mu = np.nan_to_num(this_pkmu['mu'].data)

    new_poles = []
    for tag, ell in zip(tags, ells):
        
        # compute the variance
        power = this_pkmu['power'].data
        variance = 2 * np.nansum(weights*((2*ell+1)*power*legendre(ell)(mu))**2, axis=-1) / N_1d
        
        # make the new PkResult object
        data = np.vstack([this_pole['k'], this_pole[tag], variance**0.5, this_pole['modes']]).T
        pk = pkresult.PkResult.from_dict(data, ['k', 'power', 'error', 'modes'], sum_only=['modes'], **meta)
        new_poles.append(pk)
            
    return new_poles

def format_multipoles_set(poles, pkmu, ells):
    """
    Format the input spectra set of multipoles, which includes multiple
    multipoles within one `PkResult`, such that the multipole number
    has its own dimension. Also add errors based on the specified
    `P(k,mu)` results
    """
    from . import SpectraSet

    all_data = []
    for key in poles.ndindex():

        this_pole = poles.sel(**key).values
        this_pkmu = pkmu.sel(**key).values    
        
        new_poles = format_multipoles(this_pole, this_pkmu, ells)
        all_data.append(new_poles)
        
    tags, ells = zip(*ells)
    all_data = np.reshape(all_data, poles.shape + (3,))
    coords = [poles.coords[dim] for dim in poles.dims] + [ells]
    return SpectraSet(all_data, coords, poles.dims+('ell',))
    
    
def get_valid_data(k_cen, power, kmin=-np.inf, kmax=np.inf):
    """
    Return the valid data, removing any `null` entries and
    those elements out of the specified k-range. 
    
    Notes
    -----
    If data is multi-dimensional, data must be valid across
    all higher dimensions
    
    Parameters
    ----------
    power : np.ndarray, PkmuResult, or PkResult
        the power data
    kmin : float, optional
        minimum wavenumber to trim by (inclusively), in h/Mpc
    kmax : float, optional
        maximum wavenumber to trim by (inclusively), in h/Mpc
    
    Returns
    -------
    toret : np.ndarray
        array holding the trimmed data
    """
    from nbodykit import pkresult, pkmuresult
    if isinstance(power, (pkmuresult.PkmuResult, pkresult.PkResult)):
        power = power.data.data
       
    # not null entries
    not_null = ~isnull(power, broadcast=True)    
    
    # the valid k entries
    N = np.shape(power)[-1] if np.ndim(power) > 1 else 1
    in_range = valid_k(k_cen, N, kmin=kmin, kmax=kmax)
    
    if np.ndim(power) > 1:
        idx = np.all((not_null)&(in_range), axis=-1)
    else:
        idx = (not_null)&(in_range)
    return power[idx,...]


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
        
def trim_zeros_indices(filt):
    """
    Return the indices (first, last) specifying
    the indices to trim leading or trailing zeros
    """
    first = 0
    for i in filt:
        if i != 0.:
            break
        else:
            first = first + 1
    last = len(filt)
    for i in filt[::-1]:
        if i != 0.:
            break
        else:
            last = last - 1
    return first, last

def trim_and_align_data(coords, arr, kmin=-np.inf, kmax=np.inf, k_axes=[0]):
    """
    Remove null and out-of-range entries from the input array, using
    ``k_cen`` as the k-coordinates to determine out-of-range values. 
    
    Notes
    -----
    *   If the input array has multiple dimensions, the trimmed data will 
        be `aligned` by filling with NaNs along the second dimension in 
        order to have a proper shape, in the case that the k ranges vary 
        across the second dimension.
    *   The function assumes the first two dimensions of `arr` has 
        shape ``(len(coords[0]), len(coords[1]))``, with higher dimensions
        are allowed
    
    Parameters
    ----------
    coords : list
        list of 1D arrays specifying the coordinates for the data array
    arr : numpy.ndarray
        array (possibly structured) holding the data to trim
    kmin : float, array_like, optional
        minimum wavenumber to trim by (inclusively), in h/Mpc
    kmax : float, array_like, optional
        maximum wavenumber to trim by (inclusively), in h/Mpc
    
    Returns
    -------
    coords : list
        list of new coordinates, broadcasted to the shape of data,
        with nans for missing data
    toret : numpy.ndarray
        array holding the trimmed data, which has been
        possibly re-aligned by filling with NaNs
    """ 
    k_cen = coords[0]
    
    # not null entries
    null = isnull(arr, broadcast=True)    
    
    # the valid k entries
    N = np.shape(arr)[1] if np.ndim(arr) > 1 else 1
    out_of_range = ~valid_k(k_cen, N, kmin=kmin, kmax=kmax)
    
    # remove any leading or trailing NaNs
    nan_idx = np.logical_or(null.T, out_of_range.T).T
    cnts = (~nan_idx).astype(int)
    if np.ndim(cnts) > 1: 
        axes = tuple(range(1, np.ndim(cnts)))
        cnts = np.sum(cnts, axis=axes)
    first, last = trim_zeros_indices(cnts)
    
    # do the cases of structured array or normal array
    copy = arr.copy()    
    if is_structured(arr):        
        copy[nan_idx] = tuple([np.nan]*len(copy.dtype))
    else:
        copy[nan_idx] = np.nan
    
    key = [slice(None)]*np.ndim(copy)
    for axis in k_axes:
        key[axis] = slice(first, last)
    toret = copy[key]
            
    # broadcast coords to (Nk, N)
    new_coords = np.meshgrid(*coords, indexing='ij')
    
    # remove any higher order dimensions
    for axis in reversed(range(2, np.ndim(arr))):
        nan_idx = np.take(nan_idx, 0, axis=axis)
    
    # trim and align
    for i in range(len(new_coords)):
        new_coords[i][nan_idx] = np.nan
        new_coords[i] = new_coords[i][first:last]
    return new_coords, toret

def limit_array(lim, N):
    """
    Convenience function to return array of limits
    """
    toret = np.empty(N)
    toret[:] = lim
    return toret

def valid_k(k_cen, N, kmin=-np.inf, kmax=np.inf):
    """
    Return an array of indices specifying which array
    elements are within the specified minimum and maximum
    k range
    
    Parameters
    ----------
    k_cen : (Nk,)
        a 1D aray specifying the center values of the k coordinates
    N : int
        the size of the second axis
    kmin : {float, array_like}
        a float specifying the minimum k value, or an array of size ``N``
    kmax : {float, array_like}
        a float specifying the maximum k value, or an array of size ``N``
        
    Returns
    -------
    idx : array_like
        a boolean array of size (Nk, N) specifying the valid elements
    """
    # the k limits as arrays
    kmin = limit_array(kmin, N)
    kmax = limit_array(kmax, N)
    
    # broadcast and return
    return np.squeeze((k_cen[:,None] >= kmin)&(k_cen[:,None] <= kmax))

def isnull(arr, broadcast=False):
    """
    Return a boolean array specifying the null entries of the
    input array `arr`, which uses ``pandas.isnull`` to perform
    this test
    
    Notes
    -----
    The input array can be either a normal np.ndarray or a
    structured array. If ``broadcast = True`` and the input
    array is a structured array, return a single mask, broadcasted
    across all fields
    
    Parameters
    ----------
    arr : np.ndarray
        the input array to test for null entries, which can 
        be a structured array
    broadcast : bool, optional (`False`)
        if ``True``, return a mask where each element is considered
        null if any of the structured array fields are null
    """
    import pandas as pd
    
    # structured array
    if is_structured(arr):
        
        # the return stuctured array
        dtype = zip(arr.dtype.names, [np.dtype('bool')]*len(arr.dtype))
        toret = np.empty(arr.shape, dtype=dtype)
        
        # loop over each field
        for name in arr.dtype.names:
            toret[name] = pd.isnull(arr[name])
        
        # null if any entries are null?
        if broadcast:
            toret = toret.view(bool).reshape(toret.shape + (-1,))
            toret = np.any(toret, axis=-1)
    else:
        toret = pd.isnull(arr)
    return toret
            
def is_structured(arr):
    """
    Test if the input array is a structured array
    by testing for `dtype.names`
    """
    if not isinstance(arr, np.ndarray):
        return False
    return arr.dtype.names is not None
    
def mean_structured(arr, weights=None, axis=0):
    """
    Take the mean of each field of the input recarray ``arr``, 
    optionally weighting by ``weights``, across the 
    specified axis
    
    Notes
    -----
    This will use `numpy.nansum` to take the weighted average, thus
    treating any `np.nan` elements as missing data. If the input
    array has only `np.nan` values across the sum axis, thus
    elements will be set to `np.nan` in the return array
    """
    if not is_structured(arr):
        raise ValueError("calling ``mean_structured`` on an array that is not structured")
    
    # roll desired axis to front
    if axis != 0: arr = np.rollaxis(arr, axis)
        
    # make the weights if we need to, and check
    if weights is None:
        weights = np.ones(arr.shape)
    if weights.shape != arr.shape:
        args = (str(arr.shape), str(weights.shape))
        raise ValueError("weights array should have shape %s, not %s" %args)
        
    norm = np.nansum(weights, axis=0)
    toret = np.empty(arr.shape[1:], dtype=arr.dtype)
    
    for name in arr.dtype.names:
        toret[name] = np.nansum(arr[name]*weights, axis=0)
        toret[name] /= np.nansum(arr[name]*0.+weights, axis=0)
        
        missing = np.all(np.isnan(arr[name]), axis=0)
        toret[name][missing] = np.nan
        
    return toret
