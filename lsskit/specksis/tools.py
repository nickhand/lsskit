"""
    tools.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : analysis tools
"""
from .. import numpy as np

def format_multipoles(this_pole, this_pkmu, ells):
    """
    Format the input spectra of multipoles, which includes multiple
    multipoles within one `PkResult`, such that the multipole number
    has its own dimension. Also add errors based on the specified
    `P(k,mu)` results
    """
    from scipy.special import legendre
    from nbodykit import pkresult

    tags = ells.keys()
    ells = ells.values()

    meta = {k:getattr(this_pole, k) for k in this_pole._metadata}
    meta['edges'] = this_pole.kedges
    
    # weight by modes
    modes = this_pkmu['modes'].data
    N_1d = modes.sum(axis=-1)
    weights = modes / N_1d[:,None]

    # avg mu
    mu = np.nan_to_num(this_pkmu['mu'].data)

    new_poles = []
    for tag, ell in zip(tags, ells):
        
        # compute the variance
        power = np.nan_to_num(this_pkmu['power'].data)
        variance = (weights*((2*ell+1)*power*legendre(ell)(mu))**2).sum(axis=-1) / N_1d
        
        # make the new PkResult object
        data = np.vstack([this_pole['k'], this_pole[tag], variance**0.5]).T
        pk = pkresult.PkResult.from_dict(data, ['k', 'power', 'error'], sum_only=['modes'], **meta)
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
        
    all_data = np.reshape(all_data, poles.shape + (3,))
    coords = [poles.coords[dim] for dim in poles.dims] + [ells.values()]
    return SpectraSet(all_data, coords, poles.dims+('ell',))
        


def compute_pole_covariance(power_list, ells, kmin=-np.inf, kmax=np.inf, 
                            force_diagonal=False, return_extras=False):
    """
    Compute the covariance matrix of multipole measurements, optionally 
    returning the center k and mu bins, and the mean power
    
    Parameters
    ----------
    power_list : SpectraSet
        a set of PkResult objects to compute the covariance from
    ells : list of integers
        list of multipoles numbers identifying the multipoles to concatenate
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
    
    # get the kmin/kmax as arrays
    if kmin is None: kmin = -np.inf
    if kmax is None: kmax = np.inf
    kmin_ = np.empty(len(ells))
    kmax_ = np.empty(len(ells))
    kmin_[:] = kmin
    kmax_[:] = kmax
    
    dims = list(power_list.dims)
    if 'ell' not in dims:
        raise ValueError("`ell` dimension must be present to compute pole covariance")
    dims.remove('ell')
    if len(dims) != 1:
        raise ValueError("SpectraSet must have dimension `ell` plus one other dimension")
    other_dim = dims[0]
    
    N = len(power_list[other_dim])
    data, shapes = [], []
    for i, key in enumerate(power_list[other_dim]):
        
        poles = power_list.loc[{'ell':ells, other_dim:key}]
        
        tostack = []
        for p in poles:
            p = p.values
            p.add_column('k_center', p.k_center)   
            tostack.append(p.data.copy())
        this_data = np.vstack(tostack).T
        this_data = trim_and_align_data(this_data, kmin=kmin_, kmax=kmax_)
        data.append(this_data)
        
    data = np.asarray(data)    
    
    # concatenate all ells together and remove the NaNs
    power = data['power'].reshape((N,-1), order='F')
    power = power[np.isfinite(power)].reshape((N, -1))
    
    with warnings.catch_warnings():
        mean_power = np.nanmean(data['power'], axis=0)
        k_center = np.nanmean(data['k_center'], axis=0)
        modes = np.nanmean(data['modes'], axis=0)
        
    C = np.cov(power, rowvar=False)
    if force_diagonal:
        diags = np.diag(C)
        C = np.diag(diags)
    if return_extras: 
        extras = {'mean_power':mean_power, 'modes' : modes}
        return C, k_center, extras
    else:
        return C


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
        this_data = trim_and_align_data(p.data, kmin=kmin_, kmax=kmax_)
        data.append(this_data)
        
    data = np.asarray(data)    
    
    # concatenate all ells together and remove the NaNs
    power = data['power'].reshape((N,-1), order='F')
    power = power[np.isfinite(power)].reshape((N, -1))

    with warnings.catch_warnings():
        mean_power = np.nanmean(data['power'], axis=0)
        k_center = np.nanmean(data['k_center'], axis=0)
        mu_center = np.nanmean(data['mu_center'], axis=0)
        modes = np.nanmean(data['modes'], axis=0)
        
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


def trim_and_align_data(data, kmin=None, kmax=None):
    """
    Remove any `NaN` entries and trim the input structured array
    to the specified k range. If the input array has multiple 
    dimensions, the trimmed data will be filled with NaNs to align
    it properly, in the case that the k ranges vary across
    the second dimension.
    
    Parameters
    ----------
    data : numpy.ndarray
        structured array holding the data to trim
    kmin : float, array_like, optional
        minimum wavenumber to trim by (inclusively), in h/Mpc
    kmax : float, array_like, optional
        maximum wavenumber to trim by (inclusively), in h/Mpc
    
    Returns
    -------
    toret : numpy.ndarray
        structured array holding the trimmed data, which has been
        possibly re-aligned by filling with NaNs
    """        
    columns = data.dtype.names
    shape = data.shape
    n = shape[-1] if data.ndim > 1 else 1
    # check kmin/kmax range
    if kmin is not None and not np.isscalar(kmin) and len(kmin) != n:
        raise ValueError("kmin has length %d, but should be of length %d" %(len(kmin), n))
    if kmax is not None and not np.isscalar(kmax) and len(kmax) != n:
        raise ValueError("kmax has length %d, but should be of length %d" %(len(kmax), n))
   
    # initial selection based on NaN and k ranges
    valid = np.ones(shape, dtype=bool)
    for col in columns:
        valid &= ~np.isnan(data[col])    
    if kmin is not None:
        valid &= (data['k'] >= kmin)
    if kmax is not None:
        valid &= (data['k'] <= kmax)
    
    # if multiple dimensions, align data by filling with NaNs
    if valid.ndim > 1:
        row_inds = np.arange(shape[0])
        min_idx = min([row_inds[valid[:,i]].min() for i in range(shape[1])])
        max_idx = max([row_inds[valid[:,i]].max() for i in range(shape[1])])
        nan_inds = np.zeros(shape, dtype=bool)
        nan_inds[min_idx:max_idx, :] = True
        nan_inds &= ~valid
        valid = np.zeros(shape, dtype=bool)
        valid[min_idx:max_idx, :] = True
        
    shape = list(shape)
    shape[0] = valid.sum(axis=0)
    if valid.ndim > 1:
        shape[0] = shape[0][0]
        
    # make the output
    dtype = [(name, 'f8') for name in columns]
    toret = np.empty(shape, dtype=dtype)
    for col in columns:
        if valid.ndim > 1:
            data[col][nan_inds] = np.nan
        toret[col] = data[col][valid].reshape(shape)
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
