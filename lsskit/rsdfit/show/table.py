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
        
def nonlinear_bias_header(model):
    """
    Comparison table headers for nonlinear bias model types
    """
    if model == 'b2_00_constrained_b2_01':
        return r"$b_2^{00}$, constrained $b_2^{01}$"
    elif model == 'constrained_b2_01':
        return r"constrained $b_2^{01}$"
    elif model == 'b2_01_a_b2_01_b':
        return r"$b_{2,a}^{01}$, $b_{2,b}^{01}$"
    else:
        raise ValueError("`model` value '%s' not recognized" %model)