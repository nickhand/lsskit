"""
    tools.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : tools for spectra analysis
"""

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
    columns = ['k', 'mu', 'power', 'error']
    
    valid = ~np.isnan(data['power'])
    if kmin is not None:
        valid &= (data['k'] >= kmin)
    if kmax is not None:
        valid &= (data['k'] <= kmax)
    
    toret = {}
    for col in columns:
        if col in data:
            toret[col] = data[col][valid]
        else:
            if hasattr(data, 'dtype') and hasattr(data.dtype, 'names'):
                if col in data.dtype.names:
                    toret[col] = data[col][valid]

    return toret