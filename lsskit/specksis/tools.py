"""
    tools.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : analysis tools
"""
from .. import numpy as np

def compute_pkmu_covariance(data, dk=None, kmin=-np.inf, kmax=np.inf):
    """
    Compute covariance matrix of P(k,mu) measurements
    
    Parameters
    ----------
    data : SpectraSet
        a set of PkmuResult objects to compute the covariance from
    dk : float, optional
        re-index the measurements using this k spacing
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
        
        # reindex k bins
        if dk is not None:
            p = p.reindex_k(dk, weights='modes')
            
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
