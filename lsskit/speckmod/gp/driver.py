import numpy as np
import functools
from sklearn import preprocessing
import george
import emcee
import scipy.optimize as op

#------------------------------------------------------------------------------
# mcmc GP functions
#------------------------------------------------------------------------------
def progress_report(sampler, niters, nwalkers, last=10):
    """
    Report the current status of the sampler.
    """
    k = np.count_nonzero(sampler.lnprobability[0])
    if not k:
        print("No iterations with valid parameters (chain shape = {})".format(sampler.chain.shape))
        return
    
    npars = sampler.chain.shape[-1]
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
    for i in range(npars):
        name = 'theta%d' %i
        pos = chain[:,-last:,i].ravel()
        text.append("  {:15s} = {:.6g} +/- {:<12.6g} (best={:.6g}) (autocorr: {:.3g})".format(
                    name, np.median(pos), np.std(pos), chain[best_walker, best_iter, i], acor[i]))
    
    text = "\n".join(text) +'\n'
    print(text)
    
def mcmc_lnprob(gp, y, p):
    """
    The log probality to use when fitting the GP with `emcee`
    
    Parameters
    ----------
    gp : george.GP
        the gaussian process instance 
    y : array_like
        the training data that we are fitting
    p : array_like
        the hyper-parameters array
    """
    if np.any((-10 > p) + (p > 10)):
        return -np.inf
    lnprior = 0.0
    gp.kernel.pars = np.exp(p)
    return lnprior + gp.lnlikelihood(y, quiet=True)
    
def hyperparams_from_mcmc(ns, X, Y, Yerr=None):
    """
    Find the best-fit hyperparameters, using `emcee` to do the
    likelihood maximization
    
    Parameters
    ----------
    ns : argparse.Namespace
        the namespace of arguments returned by `parse_mcmc_gp`
    X : array_like
        the independent variables of the training set
    Y : array_like
        the dependent variables of the training set
    Yerr : array_like, optional
        the error on the dependent variables of the training set. 
        If `None`, a very small fractional error is assumed for numerical purposes
    """
    # add a jitter for easier maximization
    if Yerr is None: Yerr = 1e-5*Y
        
    # default burnin is same as number of iterations
    if ns.burnin is None: ns.burnin = iterations
            
    # standardize the data
    if X.ndim == 1: X = X.reshape(-1, 1)
    X_scaler = preprocessing.StandardScaler(copy=True).fit(X)
    Y_scaler = preprocessing.StandardScaler(copy=True).fit(Y.reshape(-1, 1))
    
    # setup the kernel, optionally varying the amplitude
    ndim = X.shape[1]
    if ns.vary_amplitude:
        theta = np.ones(ndim+1)
        kernel = theta[0] * george.kernels.ExpSquaredKernel(theta[1:], ndim=X.shape[1])
    else:
        theta = np.ones(ndim)
        kernel = george.kernels.ExpSquaredKernel(theta, ndim=X.shape[1])
        
    # initialize the GP and compute once
    gp = george.GP(kernel, solver=george.HODLRSolver)
    gp.compute(X_scaler.transform(X), yerr=Yerr / Y_scaler.scale_)
    
    # setup the sampler
    y_scaled = np.squeeze(Y_scaler.transform(Y.reshape(-1,1)))
    objective = functools.partial(mcmc_lnprob, gp, y_scaled)
    sampler = emcee.EnsembleSampler(ns.nwalkers, len(kernel), objective)

    # initialize the walkers in a ball around init values
    p0 = [np.log(kernel.pars) + 1e-4 * np.random.randn(len(kernel)) for i in range(ns.nwalkers)]

    # update progress
    def update_progress(niter):
        conditions = [niter < 10, niter < 50 and niter % 2 == 0,  niter < 500 and niter % 10 == 0, niter % 100 == 0]
        return any(conditions)

    print "running burn-in"
    for i, _ in enumerate(sampler.sample(p0, iterations=ns.burnin)):
        if update_progress(i):
            progress_report(sampler, ns.burnin, ns.nwalkers)
            
    p0 = sampler.chain[:,-1,:] 
    sampler.reset()

    print "running production chain"
    for i, _ in enumerate(sampler.sample(p0, iterations=ns.iterations)):
        if update_progress(i):
            progress_report(sampler, ns.iterations, ns.nwalkers)
    
    # take the second half and average
    _, chain = np.split(sampler.chain, 2, axis=1)
    bestfit = np.exp(chain.reshape(-1, len(kernel)).mean(axis=0))
    gp.kernel.pars = bestfit
    
    print "best-fit hyperparameters = ", bestfit
    print "maximum likelihood = ", gp.lnlikelihood(y_scaled)
    
    # write the output
    np.savez(ns.output, chain=sampler.chain, bestfit=bestfit)
    
#------------------------------------------------------------------------------
# max-likelihood GP functions
#------------------------------------------------------------------------------
def nll(gp, y, p, rank=0):
    """
    The negative log-likelihood
    """
    gp.kernel.pars = np.exp(p)
    ll = gp.lnlikelihood(y, quiet=True)

    print "rank %d: parameters = " %rank, gp.kernel.pars
    print "rank %d log likelihood = " %rank, ll
    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25

def grad_nll(gp, y, p):
    """
    The gradient of the negative log-likelihood
    """
    gp.kernel.pars = np.exp(p)
    return -gp.grad_lnlikelihood(y, quiet=True)
    
    
def hyperparams_from_ml(ns, X, Y, Yerr):
    """
    Find the best-fit hyperparameters, using `scipy.optimize` to minimize
    the negative log likelihood
    
    Parameters
    ----------
    args : argparse.Namespace
        the namespace of arguments returned by `parse_mcmc_gp`
    X : array_like
        the independent variables of the training set
    Y : array_like
        the dependent variables of the training set
    Yerr : array_like, optional
        the error on the dependent variables of the training set. 
        If `None`, a very small fractional error is assumed for numerical purposes
    """
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    
    # standardize the data
    if X.ndim == 1: X = X.reshape(-1, 1)
    X_scaler = preprocessing.StandardScaler(copy=True).fit(X)
    Y_scaler = preprocessing.StandardScaler(copy=True).fit(Y.reshape(-1, 1))
    
    # setup the kernel, optionally varying the amplitude
    ndim = X.shape[1]
    if ns.vary_amplitude:
        theta = np.ones(ndim+1)
        kernel = theta[0] * george.kernels.ExpSquaredKernel(theta[1:], ndim=X.shape[1])
    else:
        theta = np.ones(ndim)
        kernel = george.kernels.ExpSquaredKernel(theta, ndim=X.shape[1])
        
    # initialize the GP 
    gp = george.GP(kernel, solver=george.HODLRSolver)
    
    y_scaled = np.squeeze(Y_scaler.transform(Y.reshape(-1,1)))
    objective = functools.partial(nll, gp, y_scaled, rank=rank)
    grad_objective = functools.partial(grad_nll, gp, y_scaled)
    
    best_lnlike = -np.inf
    best_rank = -1
    
    # run random_start fits in parallel
    done = 0
    lnlikes = []
    params = []
    for i in range(rank, ns.random_start, size):
        
        # determine the theta to start out
        if rank == 0 and done == 0 and ns.log_theta0 is not None:
            log_theta0 = ns.log_theta0
        else:
            log_theta0 = np.random.uniform(-3, 1.5, size=len(kernel))
            
        gp.kernel.pars = np.exp(log_theta0)
        print "rank %d: on random start %d, starting with " %(rank, i), gp.kernel.pars
        
        # compute once
        gp.compute(X_scaler.transform(X), yerr=Yerr / Y_scaler.scale_)
    
        # print the initial ln-likelihood.
        print "rank %d: initial log likelihood = " %rank, gp.lnlikelihood(y_scaled)

        # run the optimization routine
        p0 = gp.kernel.vector
        try:
            results = op.minimize(objective, p0, jac=grad_objective, method=ns.method, options={'disp':True})
        except:
            break

        # Update the kernel and print the final log-likelihood.
        gp.kernel.pars = np.exp(results.x)
        best_lnlike = gp.lnlikelihood(y_scaled)
        print "rank %d: final log likelihood = " %rank, best_lnlike
        print "rank %d: final parameters = "%rank, gp.kernel.pars
        lnlikes.append(best_lnlike)
        params.append(gp.kernel.pars)
        
        done += 1
    
    # put a barrier to wait for everyone
    comm.Barrier()
    
    # determine the best
    lnlikes = comm.gather(lnlikes, root=0)
    params = comm.gather(params, root=0)
    if rank == 0:
        
        lnlikes = np.concatenate(lnlikes)
        params = np.concatenate(params)
        index = lnlikes.argmax()
        
        print "maximum log likelihood = ", lnlikes[index]
        print "best parameters = ", params[index]
        
        np.savez(ns.output, lnlike=lnlikes, params=params)