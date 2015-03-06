"""
 tools.py
 lss
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 08/24/2014
"""
from . import numpy as np, tsal
import itertools
from glob import glob

#-------------------------------------------------------------------------------
def sample_by_mass_pdf(masses, masses0, bins=None, N=None):
    """
    Choose objects from the `masses` array such that the probability density
    function (pdf) of the resulting masses is consistent with the distribution
    of the `masses0` array.
    
    Parameters
    ----------
    masses : pandas.Series
        A Series holding the masses from which we will choose objects
    masses0: array_like
        The masses chosen from the desired pdf. These will be binned according
        to `bins` to compute the pdf.
    bins : optional
        The bins keyword to pass to the `np.histogram` function. If `None`, 
        use Scott's rule to determine the total number of bins
    N : int, optional
        The total number of objects to choose. If `None`, equal to the length
        of `masses0`
        
    Returns
    -------
    index : array_like
        A list of the index values of the objects chosen from `masses`. Each
        value represent the "objid" of the object
    """
    from pandas import Series
    
    if not isinstance(masses, Series):
        raise TypeError("Input masses to choose from must be an pandas.Series")
        
    # take the log10 of the masses
    logM_full = np.log10(masses)
    logM0 = np.log10(masses0)
    
    # get the bins, if not provided
    if bins is None:
        dM, bins = scotts_bin_width(logM0, return_bins=True)
    N_bins = len(bins) - 1
    
    # total number
    if N is None:
        N = len(logM0)
    
    # number counts of the desired mass distribution
    pdf, _ = np.histogram(logM0, bins=bins)
    
    # get the new counts
    new_counts = np.bincount(np.random.choice(range(N_bins), p=1.*pdf/pdf.sum(), size=N))

    # remove any objects in logM_full out of range
    inds = (logM_full >= bins[0])&(logM_full <= bins[-1])
    logM_full = logM_full[inds]

    # the bin numbers
    bin_numbers = Series(np.digitize(logM_full, bins) - 1, index=logM_full.index)

    index = []
    for bin_num, count in enumerate(new_counts):
        if count == 0:
            continue
        print bin_num
        possible = bin_numbers[bin_numbers == bin_num]
        if len(possible) == 0:
            continue
            
        if count > len(possible):
            index += list(possible.index)
        else:
            index += list(np.random.choice(possible.index, size=count))
        
    return index


#-------------------------------------------------------------------------------
def scotts_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using Scott's rule:

    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    return_bins : bool (optional)
        if True, then return the bin edges

    Returns
    -------
    width : float
        optimal bin width using Scott's rule
    bins : ndarray
        bin edges: returned if `return_bins` is True

    Notes
    -----
    The optimal bin width is

    .. math::
        \Delta_b = \frac{3.5\sigma}{n^{1/3}}

    where :math:`\sigma` is the standard deviation of the data, and
    :math:`n` is the number of data points.

    See Also
    --------
    knuth_bin_width
    freedman_bin_width
    astroML.plotting.hist
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    sigma = np.std(data)

    dx = 3.5 * sigma * 1. / (n ** (1. / 3))

    if return_bins:
        Nbins = np.ceil((data.max() - data.min()) * 1. / dx)
        Nbins = max(1, Nbins)
        bins = data.min() + dx * np.arange(Nbins + 1)
        return dx, bins
    else:
        return dx

#-------------------------------------------------------------------------------
def compute_average_biases(pattern):
    """
    Compute the average bias for a pattern specifiying a set of files
    """
    files = glob(pattern)
    return np.mean([extract_bias(f)[0] for f in files])

#-------------------------------------------------------------------------------
def add_power_labels(ax, output_units, data_type, mu_avg=False, 
                        norm_linear=True, subtract_shot_noise=True):
    """
    This will add the axis labels for power spectra
    """
    if output_units == 'relative':
        k_units = "(h/Mpc)"
        P_units = "(Mpc/h)^3"
    else:
        k_units = "(1/Mpc)"
        P_units = "(Mpc)^3"
        
    # let's set the xlabels and return
    if ax.xlabel.text == "":
        ax.xlabel.update(r"$\mathrm{k \ %s}$" %k_units)

    if ax.ylabel.text == "":

        if data_type == "Pkmu":
            if mu_avg: 
                P_label = r"\langle P(k, \mu) \rangle_\mu"
                norm_label = r"\langle P^\mathrm{EH}(k, \mu) \rangle_\mu"
            else:
                P_label = r"P(k, \mu)"
                norm_label = r"P^\mathrm{EH}(k, \mu)"
        else:
            norm_label = r"P^\mathrm{EH}_{\ell=0}(k)"
            if data_type == "monopole":
                P_label = r"P_{\ell=0}(k)"
            else:
                P_label = r"P_{\ell=2}(k)"

        if norm_linear:
            if subtract_shot_noise:
                ax.ylabel.update(r"$\mathrm{(%s - \bar{n}^{-1}) \ / \ %s}$" %(P_label, norm_label))
            else:
                ax.ylabel.update(r"$\mathrm{%s \ / \ %s}$" %(P_label, norm_label))

        else:    
            ax.ylabel.update(r"$\mathrm{%s \ %s}$" %(P_label, P_units))
#end add_power_labels

#-------------------------------------------------------------------------------
def weighted_mean(data):
    """
    Take the weighted mean of a list of `PowerMeasurement` objects
    
    Parameters
    ----------
    data : list
        The list of `PowerMeasurement` objects. Each object must have an `error`
        column in order to take the weighted average
    
    Returns
    -------
    weighted_frame : PowerMeasurement
        The DataFrame holding the weighted average of the input frames
    """
    weights = [1./d.data.variance for d in data]
    norm = np.sum(weights, axis=0)
    toret =  np.sum([d*w for d, w in zip(data, weights)]) / norm
    toret._data['variance'] /= (1.*len(data))
    return toret

#end weighted_mean

#-------------------------------------------------------------------------------
def cosmo_from_config(config_file):
    """
    Return a cosmology.Cosmology object as read from the `compute_Pkmu`
    config file, which stores the cosmology that was used
    
    Parameters
    ----------
    config_file : str
        The name of the file holding the config options
        
    Return
    ------
    cosmo : cosmology.parameters.Cosmology
    """
    from cosmology.parameters import Cosmology
    
    incosmo = {'omb' : 'omegab', \
               'omm' : 'omegac', \
               'omde' : 'omegal',\
               'omnu' : 'omegan',\
               'h'   : 'h', \
               'n_s' : 'n' , \
               'tau' : 'tau', \
               'sigma_8' : 'sigma_8'}
    
    outcosmo = {}
    for line in open(config_file, 'r'):
        try:
            fields = line[1:].split('=')
            key = fields[0].strip()
            if key in incosmo.keys():
                val = float(fields[-1].strip())
                outcosmo[incosmo[key]] = val
        except:
            continue
            
    # assume flat, and compute h
    outcosmo['omegac'] -= outcosmo['omegab']
    omegas = ['omegab', 'omegac', 'omegal', 'omegan']
    for omega in omegas: outcosmo[omega] /= outcosmo['h']**2
    
    return Cosmology(outcosmo)

#-------------------------------------------------------------------------------
def extract_bias(tsal_file):
    """
    Extract the measured linear bias from the tsal file
    """
    # now the file exists, so extract the bias
    tsal_fit = tsal.TSAL(tsal_file)
    if "bX0" not in tsal_fit.pars:
        raise ValueError("bX0 not a parameter in TSAL fit; keys are %s" %tsal_fit.pars.keys())
    bias = tsal_fit.pars["bX0"]
    
    return bias.val, bias.err
#end extract_bias

#-------------------------------------------------------------------------------
def groupby_average(frame, weighted):
    """
    Compute the weighted/unweighted average of the DataFrame columns. This
    is designed to be used with the `groupby` and `apply` functions, where the
    frame is the `DataFrame of a given group. 
    """
    has_modes = False
    if hasattr(frame, 'modes'):
        has_modes = True
        total_modes = frame.modes.sum()
        
    if weighted:
        weights = frame.modes if has_modes else 1./frame.variance
    else:
        weights = (~frame.power.isnull()).astype(float)
    weights /= np.sum(weights)
    weighted = frame.multiply(weights, axis='index')
    
    toret = np.sum(weighted, axis=0)
    toret['variance'] /= np.sum(~frame.power.isnull())
    if has_modes: toret['modes'] = total_modes
    
    return toret
#end groupby_average

#-------------------------------------------------------------------------------
