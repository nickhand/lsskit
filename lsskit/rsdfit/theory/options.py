"""
    options.py
    lsskit.rsdfit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : apply various options to the parameter set
"""

def use_mu_corr(params):
    """
    Use the mu2, mu4 corrections
    """
    params.model.correct_mu2 = True
    params.model.correct_mu4 = True
    params.options.append('mucorr')

def use_so_corr(params):
    """
    Use SO corrections in the params
    """
    # fit params
    params.sigma_so.update(vary=True, fiducial=3.)
    params.f_so.update(vary=True, fiducial=0.03)
    params.f1h_cBs.update(vary=False, fiducial=1.0)
    params.options.append('socorr')  
    
    # model params
    params.model.use_so_correction = True  

def use_vary_f1hcBs(params):
    """
    Use a varying f1h_cBs instead of socorr
    """
    # fit params
    params.sigma_so.update(vary=False, fiducial=0.)
    params.f_so.update(vary=False, fiducial=0.0)
    params.f1h_cBs.update(vary=True, fiducial=1.0)
    params.options.append('vary_f1hcBs')  
    
    # model params
    params.model.use_so_correction = False  

    
def use_free_b1cB(params):
    """
    Vary b1_cB from between b1_sA and b1_sB
    """
    params.b1_cB.update(vary=False, expr="b1_sA + gamma_b1cB * (b1_s - b1_sA)")
    params.gamma_b1cB.update(vary=True, fiducial=0.4, prior='normal', mu=0.4, sigma=0.2, min=0., max=1)
    params.options.append('free_b1cB')    

    
def use_delta_sigmas(params):
    """
    Apply nuisance factors to the higher order sigma relations
    """
    params.sigma_sA.update(vary=False, expr="delta_sigsA * sigma_s * sigmav_from_bias(sigma8_z, b1_sA) / sigmav_from_bias(sigma8_z, b1_s)")
    params.sigma_sB.update(vary=False, expr="delta_sigsB * sigma_s * sigmav_from_bias(sigma8_z, b1_sB) / sigmav_from_bias(sigma8_z, b1_s)")
    params.delta_sigsA.update(vary=True, fiducial=1., prior='normal', mu=1., sigma=0.2, min=0.)
    params.delta_sigsB.update(vary=True, fiducial=1., prior='normal', mu=1., sigma=0.2, min=0.)
    params.options.append('delta_sig')    

    
