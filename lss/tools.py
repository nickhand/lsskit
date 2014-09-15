"""
 tools.py
 lss
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 08/24/2014
"""
import numpy as np
import itertools
from . import tsal, cpm_tsal
from glob import glob

#-------------------------------------------------------------------------------
def compute_average_biases(pattern):
    """
    Compute the average bias for a pattern specifiying a set of files
    """
    files = glob(pattern)
    return np.mean([tools.extract_bias(f)[0] for f in files])

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
def weighted_average(frames):
    """
    Take the weighted average of a list of `pandas.DataFrames` holding power 
    spectra
    
    Parameters
    ----------
    frames : list
        The list of `pandas.DataFrame` objects. Each object must have an `error`
        column in order to take the weighted average
    
    Returns
    -------
    weighted_frame : pandas.DataFrame
        The DataFrame holding the weighted average of the input frames
    """
    sum_weights = sum(df.error**(-2) for df in frames)
    weighted_frame = (sum(df.div(df.error**2, axis=0) for df in frames)).div(sum_weights, axis=0)
    weighted_frame['error'] = (sum(df.error**(-2) for df in frames))**(-0.5)
    
    return weighted_frame

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