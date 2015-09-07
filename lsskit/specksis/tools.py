"""
    tools.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : analysis tools
"""
from .. import numpy as np


def compute_pkmu_covariance(power_list, kmin=-np.inf, kmax=np.inf, 
                            force_diagonal=False, return_extras=False):
    """
    Compute the covariance matrix of P(k,mu) measurements, optionally returning the 
    center k and mu bins, and the mean power
    
    Parameters
    ----------
    data : SpectraSet
        a set of PkmuResult objects to compute the covariance from
    kmin : float or array_like (`-numpy.inf`)
        the minimum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    kmax : float or array_like, (`numpy.inf`)
        the maximum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    force_diagonal : bool, optional (`False`)
        If `True`, set off-diagonal elements to zero before returning
    return_extras : bool, optional (`False`)
        If `True`, also return the center of the k/mu bins, and the mean power
    
    Returns
    -------
    covar : array_like
        the covariance matrix
    
    """
    import warnings
    
    if kmin is None: kmin = -np.inf
    if kmax is None: kmax = np.inf
    
    N = len(power_list)
    data, shapes = [], []
    for i, p in enumerate(power_list):
        p = p.values
        
        p.add_column('k_center', p.index['k_center'])
        p.add_column('mu_center', p.index['mu_center'])
        
        # get the kmin/kmax as arrays
        kmin_ = np.empty(p.Nmu)
        kmax_ = np.empty(p.Nmu)
        kmin_[:] = kmin
        kmax_[:] = kmax
            
        # get the valid entries and flatten so mus are stacked in order
        this_data = []
        for imu in range(p.Nmu):
            this_data.append(get_valid_data(p.Pk(imu), kmin=kmin_[imu], kmax=kmax_[imu]))
            if i == 0:
                shapes.append(this_data[-1].shape[0])
        this_data = np.concatenate(this_data)
        data.append(this_data)
        
    data = np.asarray(data)    
    power = data['power']
    
    def align_2d_data(arr, shapes):
        N0 = arr.shape[0]
        N = np.amax(shapes)
        
        out = np.ones((N0, N, len(shapes)))*np.nan
        for j, x in enumerate(arr):
            lower = 0
            for i, shape in enumerate(shapes):
                out[j, :shape, i] = x[lower : lower+shape]
                lower += shape
        return out
    
    with warnings.catch_warnings():
        mean_power = np.nanmean(align_2d_data(data['power'], shapes), axis=0)
        k_center = np.nanmean(align_2d_data(data['k_center'], shapes), axis=0)
        mu_center = np.nanmean(align_2d_data(data['mu_center'], shapes), axis=0)
        modes = np.nanmean(align_2d_data(data['modes'], shapes), axis=0)
        
    C = np.cov(power, rowvar=False)
    if force_diagonal:
        diags = np.diag(C)
        C = np.diag(diags)
    if return_extras: 
        extras = {'mean_power':mean_power, 'modes' : modes}
        return C, (k_center, mu_center), extras
    else:
        return C
    

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
