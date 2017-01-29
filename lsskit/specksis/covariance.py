from .. import numpy as np
from . import tools
from pyRSD.rsd import PkmuGrid, PkmuTransfer, PolesTransfer    


def _limits_from_model(kmin, kmax, model):
    """
    Internal helper function to update k limits based on model
    """
    # kmax
    if np.isscalar(kmax):
        kmax = min(kmax, model.kmax) 
    else:
        kmax[kmax > model.kmax] = model.kmax
    # kmin
    if np.isscalar(kmin):
        kmin = max(kmin, model.kmin)
    else:
        kmin[kmin < model.kmin] = model.kmin
    return kmin, kmax
    
def _return_covariance(kind, grid, **kwargs):
    """
    Internal helper function to return gaussian covariance 
    """
    kmin = kwargs['kmin']; kmax = kwargs['kmax']
    power = kwargs['power']
    
    if kind == 'pole':
        transfer = PolesTransfer(grid, kwargs['ells'], kmin=kmin, kmax=kmax, power=power)
    elif kind == 'pkmu':
        transfer = PkmuTransfer(grid, kwargs['mu_bounds'], kmin=kmin, kmax=kmax, power=power)
    else:
        raise ValueError("`kind` must be `pole` or `pkmu`")
    
    coords = transfer.coords_flat
    return transfer.to_covariance(components=kwargs['components']), coords

#------------------------------------------------------------------------------
# gaussian covariance from data measurements
#------------------------------------------------------------------------------
def data_pkmu_gausscov(pkmu, mu_bounds, kmin=-np.inf, kmax=np.inf, components=False):
    """
    Compute the gaussian covariance for a set of P(k,mu) measurements
    
    Parameters
    ----------
    pkmu : 
        the mean P(k,mu) measurement, on the finely-binned grid
    mu_bounds : array_like
        a list of tuples specifying (lower, upper) for each desired
        mu bin, i.e., [(0., 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    kmin : {float, array_like}, optional
        minimum wavenumber to trim by (inclusively), in h/Mpc
    kmax : {float, array_like}, optional
        maximum wavenumber to trim by (inclusively), in h/Mpc
    components : bool, optional
        If `True`, return only the ``mean_power`` and ``modes``
        
    Returns
    -------
    if components == False
        cov : (N, N)
            the covariance matrix for the multipole measurements
    else
        mean_power : (N, N), optional
            the mean power, cov = 2 * mean_power**2 / modes
        modes : (N, N), optional
            the number of modes
    """           
    # initialize the grid transfer
    data = pkmu.data.copy()
    grid = PkmuGrid.from_structured([pkmu.coords['k_cen'], pkmu.coords['mu_cen']], data)
    
    # return the Gaussian covariance components
    kw = {'mu_bounds':mu_bounds, 'kmin':kmin, 'kmax':kmax, 'power':data['power'], 'components':components}
    return _return_covariance('pkmu', grid, **kw)
    
def data_pole_gausscov(pkmu, ells, kmin=-np.inf, kmax=np.inf, components=False):
    """
    Compute the gaussian covariance for a set of multipole measurements,
    from the measured P(k,mu) observation
    
    Parameters
    ----------
    pkmu : 
        the mean P(k,mu) measurement, on the finely-binned grid
    ells : array_like, (Nell,)
        the desired multipole numbers
    kmin : {float, array_like}, optional
        minimum wavenumber to trim by (inclusively), in h/Mpc
    kmax : {float, array_like}, optional
        maximum wavenumber to trim by (inclusively), in h/Mpc
    components : bool, optional
        If `True`, return only the ``mean_power`` and ``modes``
        
    Returns
    -------
    if components == False
        cov : (N, N)
            the covariance matrix for the multipole measurements
    else
        mean_power : (N, N), optional
            the mean power, cov = 2 * mean_power**2 / modes
        modes : (N, N), optional
            the number of modes
    """    
    # initialize the grid transfer
    data = pkmu.data.copy()
    grid = PkmuGrid.from_structured([pkmu.coords['k_cen'], pkmu.coords['mu_cen']], data)
    
    # return the Gaussian covariance components
    kw = {'ells':ells, 'kmin':kmin, 'kmax':kmax, 'power':data['power'], 'components':components}
    return _return_covariance('pole', grid, **kw)

def cutsky_pole_gausscov(pkmu, ells, cosmo, zbins, nbar_spline, P0, fsky, kmin=-np.inf, kmax=np.inf):
    """
    Compute the gaussian covariance for a set of cutsky multipole measurements,
    from the measured P(k,mu) simulation boxes
    
    Parameters
    ----------
    pkmu : 
        the mean P(k,mu) periodic measurement, on the finely-binned grid
    ells : array_like, (Nell,)
        the desired multipole numbers
    cosmo : pygcl.Cosmology
        the cosmology instance used in the volume calculation
    zbins : array_like
        bins in redshift, used in the volume calculation
    nbar_spline : callable
        a function returning n(z)
    P0 : float
        the value of P0 used in the FKP weight
    fsky : float
        the fraction of the sky area covered
    kmin : {float, array_like}, optional
        minimum wavenumber to trim by (inclusively), in h/Mpc
    kmax : {float, array_like}, optional
        maximum wavenumber to trim by (inclusively), in h/Mpc
        
    Returns
    -------
    cov : (N, N)
        the covariance matrix for the multipole measurements
    coords : list
        list of the flat coordinates corresponding to the the cov

    """    
    # initialize the grid transfer
    data = pkmu.data.copy()
    grid = PkmuGrid.from_structured([pkmu.coords['k_cen'], pkmu.coords['mu_cen']], data)
    
    # return the Gaussian covariance components
    power = data['power'] - tools.get_Pshot(pkmu)    
    transfer = PolesTransfer(grid, ells, kmin=kmin, kmax=kmax, power=power)

    coords = transfer.coords_flat
    C = transfer.to_cutsky_covariance(cosmo, zbins, nbar_spline, P0, fsky)
    return C, coords

#------------------------------------------------------------------------------
# gaussian covariance from best-fit model
#------------------------------------------------------------------------------
def model_pkmu_gausscov(model, pkmu, mu_bounds, kmin=-np.inf, kmax=np.inf):
    """
    Compute the P(k,mu) gaussian covariance using a best-fit model to 
    evaluate P(k,mu) on a grid

    Parameters
    ----------
    model : pyRSD.rsd.GalaxySpectrum
        the model to compute P(k,mu) from
    pkmu : 
        the power instance holding the mean P(k,mu) result, which defines the
        (k,mu) grid over from which binning effects are accounted for
    mu_bounds : array_like
        a list of tuples specifying (lower, upper) for each desired
        mu bin, i.e., [(0., 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    kmin : {float, array_like}, optional
        the minimum wavenumber to include
    kmax : {float, array_like}, optional
        the maximum wavenumber to include
        
    Returns
    -------
    C : array_like
        the covariance matrix defined for the mu bins specified by `mu_bounds`
    """    
    # make the (k,mu) grid
    k = pkmu['k'].data
    mu = pkmu['mu'].data
    modes = pkmu['modes'].data
    modes[k < model.kmin] = 0.
    modes[k > model.kmax] = 0.
    grid = PkmuGrid([pkmu.coords['k_cen'], pkmu.coords['mu_cen']], k, mu, modes)

    # the model P(k,mu)
    power = model.Pgal(grid.k[grid.notnull], grid.mu[grid.notnull]) + tools.get_Pshot(pkmu)
    
    # limit the kmin/kmax to the model values
    kmin, kmax = _limits_from_model(kmin, kmax, model)
    
    # initialize the grid transfer and return the covariance
    transfer = PkmuTransfer(grid, mu_bounds, kmin=kmin, kmax=kmax, power=power)
    return transfer.to_covariance(components=False)
    
def model_pole_gausscov(model, pkmu, ells, kmin=-np.inf, kmax=np.inf):
    """
    Compute the multipole gaussian covariance using a best-fit model 
    to evaluate P(k,mu) on a grid

    Parameters
    ----------
    model : pyRSD.rsd.GalaxySpectrum
        the model to compute P(k,mu) from
    pkmu : 
        the power instance holding the mean P(k,mu) result, which defines the
        (k,mu) grid over from which binning effects are accounted for
    ells : array_like
        the list of multipole numbers
    kmin : {float, array_like}, optional
        the minimum wavenumber to include
    kmax : {float, array_like}, optional
        the maximum wavenumber to include
        
    Returns
    -------
    C : array_like
        the covariance matrix defined for multipoles specified by `ells`
    """
    # make the (k,mu) grid
    k = pkmu['k'].data
    mu = pkmu['mu'].data
    modes = pkmu['modes'].data
    modes[k < model.kmin] = 0.
    modes[k > model.kmax] = 0.
    grid = PkmuGrid([pkmu.coords['k_cen'], pkmu.coords['mu_cen']], k, mu, modes)

    # the model P(k,mu)
    power = model.Pgal(grid.k[grid.notnull], grid.mu[grid.notnull]) + tools.get_Pshot(pkmu)
    
    # limit the kmin/kmax to the model values
    kmin, kmax = _limits_from_model(kmin, kmax, model)
    
    # return the Gaussian covariance components
    kw = {'ells':ells, 'kmin':kmin, 'kmax':kmax, 'power':power, 'components':False}
    return _return_covariance('pole', grid, **kw)

#------------------------------------------------------------------------------
# covariance matrix from data
#------------------------------------------------------------------------------
def _covariance_from_data(coords, arr, kmin, kmax):
    """
    Internal function to compute the covariance matrix from a set 
    of power observations
    """
    # trim and align
    new_coords, arr = tools.trim_and_align_data(coords, arr, kmin=kmin, kmax=kmax)
    sizes = np.isfinite(new_coords[0]).sum(axis=0)
    new_coords = list(map(tools.flat_and_nonnull, new_coords))
    mean_arr = tools.mean_structured(arr, axis=-1) # last axis is diff realizations
    
    # flatten the power and remove the NaNs
    N = arr.shape[-1]
    power = np.rollaxis(arr['power'], 2).reshape((N, -1), order='F')
    power = power[np.isfinite(power)].reshape((N, -1))

    # take the mean for extra info
    modes = None
    mean_power = np.nanmean(arr['power'], axis=-1)
    if 'modes' in arr.dtype.names:
        modes = np.nanmean(arr['modes'], axis=-1)

    # covariance matrix and optionally, force diagonal
    C = np.cov(power, rowvar=False)
    return C, new_coords, sizes, {'mean_power':mean_power, 'modes':modes}


def compute_pole_covariance(pole_set, 
                            ells, 
                            kmin=-np.inf, 
                            kmax=np.inf, 
                            force_diagonal=False,
                            extras=False):
    """
    Compute the covariance matrix of multipole measurements, optionally 
    returning the center k and mu bins, and the mean power
    
    Parameters
    ----------
    pole_set : SpectraSet
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
    extras : bool, optional (`False`)
        If `True`, also return the center of the k/ell bins, and the mean power
    
    Returns
    -------
    covar : array_like
        the covariance matrix
    (k_cen, ell_cen) : tuple
        tuple of the k_cen and ell_cen coordinates after trimming
    extras : dict
        dict holding the `mean_power` and `modes`
    """
    ells = np.asarray(ells, dtype=float)
    
    # default limits
    if kmin is None: kmin = -np.inf
    if kmax is None: kmax = np.inf
    
    # first stack the structured arrays
    arr = tools.stack_multipoles(pole_set, ells=ells)
     
    # get the coordinates from the first box
    pole_0 = pole_set.isel(**{k:0 for k in pole_set.dims}).get()
    coords = [pole_0.coords['k_cen'], np.asarray(ells, dtype=float)]
    
    # do the work
    C, new_coords, sizes, x = _covariance_from_data(coords, arr, kmin, kmax)
    if force_diagonal:
        for i in range(len(ells)):
            for j in range(i, len(ells)):
                remove_pole_off_diags(C, i, j, sizes)
    
    # return
    return (C, new_coords, x) if extras else C

def compute_pkmu_covariance(pkmu_set, 
                            kmin=-np.inf, 
                            kmax=np.inf, 
                            force_diagonal=False, 
                            extras=False):
    """
    Compute the covariance matrix of P(k,mu) measurements, optionally 
    returning the center k and mu bins, and the mean power
    
    Parameters
    ----------
    pkmu_set : SpectraSet
        a set of PkmuResult objects to compute the covariance from
    kmin : float or array_like (`-numpy.inf`)
        the minimum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    kmax : float or array_like, (`numpy.inf`)
        the maximum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    force_diagonal : bool, optional (`False`)
        If `True`, set off-diagonal elements to zero before returning
    extras : bool, optional (`False`)
        If `True`, also return the center of the k/mu bins, and the mean power
    
    Returns
    -------
    covar : array_like
        the covariance matrix
    (k_cen, mu_cen) : tuple
        tuple of the k_cen and mu_cen coordinates after trimming
    extras : dict
        dict holding the `mean_power` and `modes`
    """
    # default limits
    if kmin is None: kmin = -np.inf
    if kmax is None: kmax = np.inf
    
    # first stack the structured arrays
    arr = np.asarray([x.values.data for x in pkmu_set])
    arr = np.rollaxis(arr, 0, arr.ndim)
     
    # get the coordinates from the first box
    pkmu_0 = pkmu_set[0].get()
    coords = [pkmu_0.coords['k_cen'], pkmu_0.coords['mu_cen']]
    
    # do the work
    C, new_coords, sizes, x = _covariance_from_data(coords, arr, kmin, kmax)
    if force_diagonal:
        diags = np.diag(C)
        C = np.diag(diags)
    
    # return
    return (C, new_coords, x) if extras else C

def remove_pole_off_diags(C, i, j, shapes):
    """
    From a covariance matrix of multipole measurements, set
    the non-diagonal elements of each sub-matrix 
    """
    inds = np.concatenate([[0], shapes.cumsum()])
    sl_i = slice(inds[i], inds[i+1])
    sl_j = slice(inds[j], inds[j+1])
    C[sl_i, sl_j] = np.diag(np.diag(C[sl_i, sl_j]))
    if i != j:
        C[sl_j, sl_i] = C[sl_i, sl_j]