"""
 tools.py
 lss
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 08/24/2014
"""
import numpy as np
import itertools
import tsal
from . import tsal, cpm_tsal

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
def extract_multipoles(tsal_file):
    """
    Extract the multipoles (monopole/quadrupole) from a TSAL file
    
    Parameters
    ----------
    tsal_file : str 
        The name of the file holding this TaylorSerierApproximatedLikelihood
        for the multipoles fit
        
    Returns
    -------
    ks : np.ndarray
        The array of wavenumbers where the multipoles are defined
    monopole : np.ndarray
        The monopole moment values
    monopole_err : np.narray
        The errors on the monopole
    quadrupole : np.ndarray
        The quadrpole moment values
    quadrupole_err : np.narray
        The errors on the quadrupole    
    """
    # read in the tsal file
    tsal_fit = tsal.TSAL(tsal_file)
    
    mono, quad = {}, {}
    for key, val in tsal_fit.pars.iteritems():
    
        k = float(key.split('_')[-1])
        if 'mono' in key:
            mono[k] = (val.val, val.err)
        elif 'quad' in key:
            quad[k] = (val.val, val.err)
            
    ks = np.array(sorted(mono.keys()))
    mono_vals, mono_errs = map(np.array, zip(*[mono[k] for k in ks]))
    quad_vals, quad_errs = map(np.array, zip(*[quad[k] for k in ks]))
    return ks, mono_vals, mono_errs, quad_vals, quad_errs
#end extract_multipoles

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
def extract_Pkmu_data(tsal_file):
    """
    Extract the P(k, mu) given the output file TSAL file holding the 
    power spectrum measurement from the ``measure_and_fit_discrete.out`` code.
    
    Return a ``pandas.DataFrame`` holding the power spectrum data
    """
    import pandas as pd
    
    # read in the measurement
    data = cpm_tsal.CPM_TSAL(tsal_file)
    
    # all combinations of (mu, k)
    muks = list(itertools.product(sorted(data.mus), sorted(data.ks)))
    
    # the column values for each (mu, k)
    columns = [data.getMeasurement(k, mu) for (mu, k) in muks]
    
    # now make the DataFrame
    index = pd.MultiIndex.from_tuples(muks, names=['mu', 'k'])
    frame = pd.DataFrame(columns, index=index, columns=['power', 'error', 'noise', 'baseline'])
    
    return frame
#end extract_Pkmu_data

#-------------------------------------------------------------------------------