def kmax_header(kmax):
    """
    Comparison table headers for different kmax
    """
    name_base = r"$k_\mathrm{max} = %s$ $h/\mathrm{Mpc}$"
    return name_base %(kmax[0]+"."+kmax[1])

def solver_header(solver):
    """
    Comparison table headers for different solvers
    """
    if (solver == 'nlopt'):
        return 'lbfgs'
    else:
        return 'emcee'
    
def fixed_alphas_header(model):
    """
    Comparison table headers for different model types
    """
    if (model == 'base'):
        return r"varying $\alpha_\perp$, $\alpha_\parallel$"
    else:
        return r"fixed $\alpha_\perp$, $\alpha_\parallel$"