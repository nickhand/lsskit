import numpy as np
from scipy.special import legendre
    
def compute_nonuniform_mu_edges(lmax):
    """
    Return the edges of non-uniform mu bins, designed
    to cancel a mu=0 systematic
    """
    def bins(lmax):
        grand, _ = legendre(lmax+1) / np.poly1d((1,0))
        indef = np.polyint(grand)
        indef -= indef(1)
        roots = np.roots(indef)
        posreals = roots > 0
        roots = roots[posreals].real
        roots.sort()
        assert len(roots) == lmax//2 + 1
        return roots
    
    upper = bins(lmax)
    nbins = len(upper)
    lower = np.append(0, upper[:-1])
    width = upper - lower
    edges = np.append(0, upper)
    return edges
    
def to_pkmu(poles, mu_edges, max_ell, attrs={}):
    """
    Invert the measured multipoles :math:`P_\ell(k)` into power
    spectrum wedges, :math:`P(k,\mu)`
    
    Parameters
    ----------
    mu_edges : array_like
        the edges of the :math:`\mu` bins
    max_ell : int
        the maximum multipole to use when computing the wedges; 
        all even multipoles with :math:`ell` less than or equal
        to this number are included
    
    Returns
    -------
    pkmu : DataSet
        a data set holding the :math:`P(k,\mu)` wedges
    """
    from scipy.integrate import quad
    from nbodykit.dataset import DataSet
    
    def compute_coefficient(ell, mumin, mumax):
        """
        Compute how much each multipole contributes to a given wedges.
        This returns:
        
        .. math::
            \frac{1}{\mu_{max} - \mu_{max}} \int_{\mu_{min}}^{\mu^{max}} \mathcal{L}_\ell(\mu)
        """
        norm = 1.0 / (mumax - mumin)
        return norm * quad(lambda mu: legendre(ell)(mu), mumin, mumax)[0]
    
    # make sure we have all the poles measured
    ells = list(range(0, max_ell+1, 2))
    if any('power_%d' %ell not in poles for ell in ells):
        raise ValueError("measurements for ells=%s required if max_ell=%d" %(ells, max_ell))
    
    # new data array
    dtype = np.dtype([('power', 'c8'), ('k', 'f8'), ('mu', 'f8')])
    data = np.zeros((poles.shape[0], len(mu_edges)-1), dtype=dtype)
    
    # loop over each wedge
    bounds = list(zip(mu_edges[:-1], mu_edges[1:]))
    for imu, mulims in enumerate(bounds):
        
        # add the contribution from each Pell
        for ell in ells:
            coeff = compute_coefficient(ell, *mulims)
            data['power'][:,imu] += coeff * poles['power_%d' %ell]
            
        data['k'][:,imu] = poles['k']
        data['mu'][:,imu] = np.ones(len(data))*0.5*(mulims[1]+mulims[0])
        
    dims = ['k', 'mu']
    edges = [poles.edges['k'], mu_edges]
    return DataSet(dims=dims, edges=edges, data=data, **attrs)