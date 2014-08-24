"""
 tools.py
 lss
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 08/24/2014
"""
#-------------------------------------------------------------------------------
def extract_multipoles(tsal_file):
    """
    Extract the multipoles (monopole/quadrupole) from a TSAL file
    """
    import tsal
    
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
    import tsal
        
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
    import itertools
    
    # read in the measurement
    data = ComovingPowerMeasurement(tsal_file)
    
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