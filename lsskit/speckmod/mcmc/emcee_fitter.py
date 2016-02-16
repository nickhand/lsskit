import numpy as np, os
import emcee
import time
import logging
from pyRSD.rsdfit.results import EmceeResults

logger = logging.getLogger('emcee_fitter')
logger.addHandler(logging.NullHandler())

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def prior_init(theory, nwalkers):
    """
    Initialize variables from univariate prior
    
    Parameters
    ----------
    model : lmfit.Model
        model object which has a `bounds` attribute giving the min/max allowed
        value for each parameter and a `free_param_names` attribute giving the
        name of the free parameters
    nwalkers : int
        the number of walkers used in fitting with `emcee`
        
    Returns
    -------
    toret : numpy.ndarray
        the initial set of parameters drawn uniformly from the prior; the shape
        is (nwalkers, npar)

    """
    # get the free params
    pars = theory.free
    
    # draw func
    draw_func = 'get_value_from_prior'

    # create an initial set of parameters from the priors (shape: nwalkers x npar)
    return np.array([getattr(par, draw_func)(size=nwalkers) for par in pars]).T
    
#-------------------------------------------------------------------------------
def update_progress(free_pars, sampler, niters, nwalkers, last=10):
    """
    Report the current status of the sampler.
    """
    k = sum(sampler.lnprobability[0] < 0)
    if not k:
        logger.warning("No iterations with valid parameters (chain shape = {})".format(sampler.chain.shape))
        return None
        
    chain = sampler.chain[:,:k,:]
    logprobs = sampler.lnprobability[:,:k]
    best_iter = np.argmax(logprobs.max(axis=0))
    best_walker = np.argmax(logprobs[:,best_iter])
    text = ["EMCEE Iteration {:>6d}/{:<6d}: {} walkers, {} parameters".format(k-1, niters, nwalkers, chain.shape[-1])]
    text+= ["      best logp = {:.6g} (reached at iter {}, walker {})".format(logprobs.max(axis=0)[best_iter], best_iter, best_walker)]
    try:
        acc_frac = sampler.acceptance_fraction
        acor = sampler.acor
    except:
        acc_frac = np.array([np.nan])        
    text += ["      acceptance_fraction ({}->{} (median {}))".format(acc_frac.min(), acc_frac.max(), np.median(acc_frac))]      
    for i, name in enumerate(free_pars):
        pos = chain[:,-last:,i].ravel()
        text.append("  {:15s} = {:.6g} +/- {:<12.6g} (best={:.6g}) (autocorr: {:.3g})".format(
                    name, np.median(pos), np.std(pos), chain[best_walker, best_iter, i], acor[i]))
    
    text = "\n".join(text) +'\n'
    logger.warning(text)

#-------------------------------------------------------------------------------
def solve(params, theory, objective, init_values=None):
    """
    Perform MCMC sampling of the parameter space using `emcee`.
    """        
    # get params
    nwalkers  = params['walkers'].value
    niters    = params['iterations'].value
    init_from = params['init_from'].value
    free_pars = theory.free_names
    ndim      = len(free_pars)
    
    #---------------------------------------------------------------------------
    # let's check a few things so we dont mess up too badly
    #---------------------------------------------------------------------------
    
    # now, if the number of walkers is smaller then twice the number of
    # parameters, adjust that number to the required minimum and raise a warning
    if 2*ndim > nwalkers:
        logger.warning("EMCEE: number of walkers ({}) cannot be smaller than 2 x npars: set to {}".format(nwalkers, 2*ndim))
        nwalkers = 2*ndim
    
    if nwalkers % 2 != 0:
        nwalkers += 1
        logger.warning("EMCEE: number of walkers must be even: set to {}".format(nwalkers))
    
    #---------------------------------------------------------------------------
    # initialize the parameters
    #---------------------------------------------------------------------------
    # 1) initialixe from initial provided values
    if init_from == 'fiducial' or init_from == 'chain':
        if init_values is None:
            raise ValueError("EMCEE: cannot initialize around best guess -- none provided")
        
        # initialize in random ball
        # shape is (nwalkers, ndim)
        p0 = np.array([init_values + 1e-3*np.random.randn(ndim) for i in range(nwalkers)])
        logger.warning("EMCEE: initializing walkers in random ball around best guess parameters")

    # 3) init from prior
    else:

        logger.warning("Attempting univariate initialization from priors")
        p0 = prior_init(theory, nwalkers)
        logger.warning("Initialized walkers from priors with univariate distributions")
            
    # initialize the sampler
    logger.warning("EMCEE: initializing sampler with {} walkers".format(nwalkers))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, objective)

    # iterator interface allows us to trap ctrl+c and know where we are
    exception = False
    burnin = params.get('burnin', 0)
    try:                               
        logger.warning("EMCEE: running {} iterations with {} free parameter(s)...".format(niters, ndim))
        start = time.time()    
        generator = sampler.sample(p0, iterations=niters, storechain=True)

        # loop over all the steps
        for niter, result in enumerate(generator):                    
            if niter < 10:
                update_progress(free_pars, sampler, niters, nwalkers)
            elif niter < 50 and niter % 2 == 0:
                update_progress(free_pars, sampler, niters, nwalkers)
            elif niter < 500 and niter % 10 == 0:
                update_progress(free_pars, sampler, niters, nwalkers)
            elif niter % 100 == 0:
                update_progress(free_pars, sampler, niters, nwalkers)

        stop = time.time()
        logger.warning("EMCEE: ...iterations finished. Time elapsed: {}".format(hms_string(stop-start)))

        # acceptance fraction should be between 0.2 and 0.5 (rule of thumb)
        logger.warning("EMCEE: mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        try:
            logger.warning("EMCEE: autocorrelation time: {}".format(sampler.get_autocorr_time()))
        except:
            pass
        
    except KeyboardInterrupt:
        exception = True
        logger.warning("EMCEE: ctrl+c pressed - saving current state of chain")
    finally:
        result = EmceeResults(sampler, theory, burnin=burnin)
        
    return result
    
