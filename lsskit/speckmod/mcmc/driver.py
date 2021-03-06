import os
import logging
import functools
import itertools
import numpy as np
from pyRSD.rsdfit.parameters import ParameterSet
from . import emcee_fitter
from .theory import lnprob

def setup_mcmc(param_file):
    """
    Setup the mcmc run by reading the parameters and 
    initialize the theory object
    
    Parameters
    ----------
    param_file : str
        the name of the file holding the mcmc parameters
    
    Returns
    -------
    params : ParameterSet
        the `driver` mcmc parameters
    theory_params : ParameterSet
        the `theory` mcmc parameters
    init_values : array_like
        the initial parameter values to use
    """
    # add a console logger
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')
    
    # first load the parameters from file
    params = ParameterSet.from_file(param_file, tags=['driver', 'theory'])
    
    # theory params
    theory_params = params['theory']
    theory_params.prepare_params()
    
    # fiducial init?
    init_values = None
    if params['driver']['init_from'] == 'fiducial':
        init_values = theory_params.free_values
    
    return params['driver'], theory_params, init_values
    
def run_binned_mcmc(args, dims, coords, theory_model, data_loader, 
                    meta=[], return_eval_kws=False):
    """
    A generator that runs the mcmc solver over a specified 
    coordinate grid of bins
    
    Parameters
    ----------
    args : argparse.Namespace
        the namespace of arguments returned by `parse_binned_mcmc`, which
        should have an attribute for each value in `dims` and a `param_file`
    dims : list
        the list of the dimension names for the coordinate grid
    coords : list
        the list of the coordinate values corresponding to the individual bins,
        with names set by `dims`
    theory_model : lmfit.Model
        the model class that will be called to return the theoretical model
    data_loader : callable
        a function that should take a dictionary specifying the coordinate
        values as the only argument
    meta : list, optional
        list of options to return from the data keywords
    return_eval_kws : bool, optional
        return the dictionary of arrays used for model evaluation
        
    Returns
    -------
    key : dict
        the dimension and coordinate values for each bin being iterated over
    result : EmceeResults
        the result of the mcmc run, stored as an `EmceeResults` instance
    extra : dict
        dictionary of extra info to return for each bin
    """
    # determine the bins we want to run
    bins = []
    for i, dim in enumerate(dims):
        val = getattr(args, dim)
        if len(val) == 1 and val[0] == 'all':
            val = coords[i]
        bins.append(val)

    # setup the mcmc run
    params, theory_params, init_values = setup_mcmc(args.param_file)
    
    # loop over each bin coordinate
    for b in itertools.product(*bins):
        
        # load
        key = dict(zip(dims, b))
        print("processing bin: %s..." %(", ".join("%s = %s" %(k, key[k]) for k in dims)))
        data_kws = data_loader(key)
    
        # make the objective function
        objective = functools.partial(lnprob, model=theory_model, theory=theory_params, **data_kws)

        # run emcee
        result = emcee_fitter.solve(params, theory_params, objective, init_values=init_values)
        print(result)
        
        # also return some meta information
        extra = {k:data_kws[k] for k in meta}
        if not return_eval_kws:
            yield key, result, extra
        else:
            yield key, result, extra, data_kws

def run_global_mcmc(args, theory_model, data_loader):
    """
    A generator that runs the mcmc solver over a specified 
    coordinate grid of bins
    
    Parameters
    ----------
    args : argparse.Namespace
        the namespace of arguments returned by `parse_global_mcmc`
    theory_model : lmfit.Model
        the model class that will be called to return the theoretical model
    data_loader : callable
        a function that should take a dictionary specifying the coordinate
        values as the only argument
        
    Returns
    -------
    result : EmceeResults
        the result of the mcmc run, stored as an `EmceeResults` instance
    """
    # setup the mcmc run
    params, theory_params, init_values = setup_mcmc(args.param_file)
    
    # check for init_from == chain
    if params['init_from'] == 'chain':
        from pyRSD.rsdfit.results import EmceeResults
        r = EmceeResults.from_npz(params['start_chain'].value)
        best = dict(zip(r.free_names, r.max_lnprob_values()))
        init_values = np.array([best[k] for k in theory_params.free_names])
    
    # load the data
    data_kws = data_loader()

    # make the objective function
    objective = functools.partial(lnprob, model=theory_model, theory=theory_params, **data_kws)

    # run emcee
    result = emcee_fitter.solve(params, theory_params, objective, init_values=init_values)
    print(result)
    
    return result
    
    
