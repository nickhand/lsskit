import numpy as np
import logging

def set_free_parameters(theory, theta):
    """
    Given an array of values `theta`, set the free parameters
    """
    # set the parameters
    for val, name in zip(theta, theory.free_names):
        theory[name].value = val
               
    # only do this if the free params have finite priors
    if any(not np.isfinite(theory[k].lnprior) for k in theory.free_names):
        return False

    #try to update
    try:
        theory.update_values()
    except Exception as e:
        args = (str(e), str(theory))
        msg = "error trying to update fit parameters; original message:\n%s\n\ncurrent parameters:\n%s" %args
        raise RuntimeError(msg)
    return True


def lnlike(theta, y=None, yerr=None, model=None, theory=None, **kwargs):
    """
    The log of the likelihood, equal to -0.5 * chisq
    """      
    # pass the model parameters as keywords
    for k in model.param_names:
        kwargs[k] = theory[k].value
    
    # model eval
    mu = model.eval(**kwargs)
    
    chi2 = (y-mu)
    if yerr is not None:
        chi2 /= yerr        
    return -0.5 * np.sum(chi2**2)


def lnprob(theta, y=None, yerr=None, model=None, theory=None, **kwargs):
    """
    Return the log of the posterior probability function (to within a constant), 
    defined as likelihood * prior.
    
    This returns:
    
    ..math: -0.5 * chi2 + lnprior 
    """
    # set the free parameters
    good_model = set_free_parameters(theory, theta)
    if not good_model: return -np.inf
            
    lp = sum(param.lnprior for param in theory.free)
    if not np.isfinite(lp):
        return -np.inf
    else:
        toret = lp + lnlike(theta, y=y, yerr=yerr, model=model, theory=theory, **kwargs) 
        return toret if not np.isnan(toret) else -np.inf
