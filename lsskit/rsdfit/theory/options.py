"""
    options.py
    lsskit.rsdfit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : apply various options to the parameter set
"""
from .. import AttrDict
valid_theory_options = ['vary_sigmav', 'mu_corr', 'so_corr', 'vary_f1hcBs', 'free_b1cB', 'delta_sigmas', 'fixed_alphas']
valid_theory_options += ['b2_00', 'b2_00_a', 'b2_00_b', 'b2_00_c', 'b2_00_d', 'b2_01_a', 'b2_01_b', 'constrained_b2_01']

def use_b2_00(params):
    """
    Add the polynomal coeffecients for the nonlinear bias 
    ``b2_00`` as parameters
    """
    name = 'b2_00'
    biases_params = [name + '_%s' %i for i in ['a', 'b', 'c', 'd']] + [name + '__' + str(i) for i in [0, 2, 4]]
    
    # make the parameters valid
    for b in biases_params:
        if b not in params.valid_params:
            params.valid_params.append(b)
        
    params.b2_00__0 = AttrDict(vary=True, fiducial=-0.6, prior='normal', mu=-0.6, sigma=0.2)
    params.b2_00__2 = AttrDict(vary=True, fiducial=0.03, prior='normal', mu=0.03, sigma=0.1)
    params.b2_00__4 = AttrDict(vary=True, fiducial=0.05, prior='normal', mu=0.05, sigma=0.05)
        
    params.b2_00_a = AttrDict(vary=False, expr="[b2_00__4, 0., b2_00__2, 0., b2_00__0]")
    params.b2_00_b = AttrDict(vary=False, expr="b2_00_a")
    params.b2_00_c = AttrDict(vary=False, expr="b2_00_a")
    params.b2_00_d = AttrDict(vary=False, expr="b2_00_a")
    
    params.options.append('b2_00')


def use_b2_00_a(params):
    """
    Add the polynomal coeffecients for the nonlinear bias 
    ``b2_00_a`` as parameters
    """
    name = 'b2_00_a'
    biases_params = [name] + [name + '__' + str(i) for i in [0, 2, 4]]
    
    # make the parameters valid
    for b in biases_params:
        if b not in params.valid_params:
            params.valid_params.append(b)
        
    params.b2_00_a__0 = AttrDict(vary=True, fiducial=-0.91, prior='normal', mu=-0.91, sigma=0.2)
    params.b2_00_a__2 = AttrDict(vary=True, fiducial=0.42, prior='normal', mu=0.42, sigma=0.15)
    params.b2_00_a__4 = AttrDict(vary=True, fiducial=0.008, prior='normal', mu=0.008, sigma=0.05)
    params.b2_00_a    = AttrDict(vary=False, expr="[b2_00_a__4, 0., b2_00_a__2, 0., b2_00_a__0]")
    
    params.options.append('b2_00_a')
    
def use_b2_00_b(params):
    """
    Add the polynomal coeffecients for the nonlinear bias 
    ``b2_00_b`` as parameters
    """
    name = 'b2_00_b'
    biases_params = [name] + [name + '__' + str(i) for i in [0, 2, 4]]
    
    # make the parameters valid
    for b in biases_params:
        if b not in params.valid_params:
            params.valid_params.append(b)
        
    params.b2_00_b__0 = AttrDict(vary=True, fiducial=-0.63, prior='normal', mu=-0.63, sigma=0.2)
    params.b2_00_b__2 = AttrDict(vary=True, fiducial=0.32, prior='normal', mu=0.32, sigma=0.15)
    params.b2_00_b__4 = AttrDict(vary=True, fiducial=0.03, prior='normal', mu=0.03, sigma=0.05)
    params.b2_00_b    = AttrDict(vary=False, expr="[b2_00_b__4, 0., b2_00_b__2, 0., b2_00_b__0]")
    
    params.options.append('b2_00_b')
    
def use_b2_00_c(params):
    """
    Add the polynomal coeffecients for the nonlinear bias 
    ``b2_00_c`` as parameters
    """
    name = 'b2_00_c'
    biases_params = [name] + [name + '__' + str(i) for i in [0, 2, 4]]
    
    # make the parameters valid
    for b in biases_params:
        if b not in params.valid_params:
            params.valid_params.append(b)
        
    params.b2_00_c__0 = AttrDict(vary=True, fiducial=-0.6, prior='normal', mu=-0.6, sigma=0.2)
    params.b2_00_c__2 = AttrDict(vary=True, fiducial=0.03, prior='normal', mu=0.03, sigma=0.1)
    params.b2_00_c__4 = AttrDict(vary=True, fiducial=0.05, prior='normal', mu=0.05, sigma=0.05)
    params.b2_00_c    = AttrDict(vary=False, expr="[b2_00_c__4, 0., b2_00_c__2, 0., b2_00_c__0]")
    
    params.options.append('b2_00_c')
    
def use_b2_00_d(params):
    """
    Add the polynomal coeffecients for the nonlinear bias 
    ``b2_00_d`` as parameters
    """
    name = 'b2_00_d'
    biases_params = [name] + [name + '__' + str(i) for i in [0, 2, 4]]
    
    # make the parameters valid
    for b in biases_params:
        if b not in params.valid_params:
            params.valid_params.append(b)
        
    params.b2_00_d__0 = AttrDict(vary=True, fiducial=0.47, prior='normal', mu=0.47, sigma=0.2)
    params.b2_00_d__2 = AttrDict(vary=True, fiducial=-0.5, prior='normal', mu=-0.5, sigma=0.2)
    params.b2_00_d__4 = AttrDict(vary=True, fiducial=0.10, prior='normal', mu=0.10, sigma=0.05)
    params.b2_00_d    = AttrDict(vary=False, expr="[b2_00_d__4, 0., b2_00_d__2, 0., b2_00_d__0]")
    
    params.options.append('b2_00_d')
    
def use_b2_01_a(params):
    """
    Add the polynomal coeffecients for the nonlinear bias 
    ``b2_01_a`` as parameters
    """
    name = 'b2_01_a'
    biases_params = [name] + [name + '__' + str(i) for i in [0, 1, 2]]
    
    # make the parameters valid
    for b in biases_params:
        if b not in params.valid_params:
            params.valid_params.append(b)
        
    params.b2_01_a__0 = AttrDict(vary=True, fiducial=0.45, prior='normal', mu=0.45, sigma=0.2)
    params.b2_01_a__1 = AttrDict(vary=True, fiducial=-1.5, prior='normal', mu=-1.5, sigma=0.5)
    params.b2_01_a__2 = AttrDict(vary=True, fiducial=0.7, prior='normal', mu=0.7, sigma=0.2)
    params.b2_01_a    = AttrDict(vary=False, expr="[0., 0., b2_01_a__2, b2_01_a__1, b2_01_a__0]")
    
    params.options.append('b2_01_a')

def use_constrained_b2_01(params):
    """
    Add the polynomal coeffecients for the nonlinear bias 
    ``b2_01_b`` as parameters, fixing them to be the same as ``b2_01_a``
    """
    # make we set up b2_01_a
    if 'b2_01_a' not in params.valid_params:
        use_b2_01_a(params)
        if 'b2_01_a' in params.options:
            params.options.pop(params.options.index('b2_01_a'))
        
    name = 'b2_01_b'
    biases_params = [name] + [name + '__' + str(i) for i in [0, 1, 2]]
    
    # make the parameters valid
    for b in biases_params:
        if b not in params.valid_params:
            params.valid_params.append(b)
        
    params.b2_01_b__0 = AttrDict(vary=True, fiducial=1.9, prior='normal', mu=1.9, sigma=0.6)
    params.b2_01_b__1 = AttrDict(vary=False, expr="b2_01_a__1")
    params.b2_01_b__2 = AttrDict(vary=False, expr="b2_01_a__2")
    params.b2_01_b    = AttrDict(vary=False, expr="[0., 0., b2_01_b__2, b2_01_b__1, b2_01_b__0]")
    params.options.append('constrained_b2_01')
    
def use_b2_01_b(params):
    """
    Add the polynomal coeffecients for the nonlinear bias 
    ``b2_01_b`` as parameters
    """
    name = 'b2_01_b'
    biases_params = [name] + [name + '__' + str(i) for i in [0, 1, 2]]
    
    # make the parameters valid
    for b in biases_params:
        params.valid_params.append(b)
        
    params.b2_01_b__0 = AttrDict(vary=True, fiducial=1.9, prior='normal', mu=1.9, sigma=0.6)
    params.b2_01_b__1 = AttrDict(vary=True, fiducial=-1.9, prior='normal', mu=-1.9, sigma=0.6)
    params.b2_01_b__2 = AttrDict(vary=True, fiducial=0.9, prior='normal', mu=0.9, sigma=0.3)
    params.b2_01_b    = AttrDict(vary=False, expr="[0., 0., b2_01_b__2, b2_01_b__1, b2_01_b__0]")
    
    params.options.append('b2_01_b')
    
def use_vary_sigmav(params):
    """
    Vary the velocity dispersion, ``sigma_v``
    """
    params.valid_params.append('sigma_v')
    params.sigma_v = AttrDict(vary=True, fiducial=4.5, prior='normal', mu=4.5, sigma=0.3)
    params.model.vel_disp_from_sims = False
    params.options.append('vary_sigmav')
    
def use_fixed_alphas(params):
    """
    Set alphas to 1 and do not vary
    """
    params.alpha_perp.update(vary=False, fiducial=1.0)
    params.alpha_par.update(vary=False, fiducial=1.0)
    params.options.append('fixed_alphas')
    
def use_mu_corr(params):
    """
    Use the mu2, mu4 corrections
    """
    params.model.correct_mu2 = True
    params.model.correct_mu4 = True
    params.options.append('mucorr')
    
def use_mu2_corr(params):
    """
    Use the mu2 corrections
    """
    params.model.correct_mu2 = True
    params.model.correct_mu4 = False
    params.options.append('mu2corr')
    
def use_mu4_corr(params):
    """
    Use the mu4 corrections
    """
    params.model.correct_mu2 = False
    params.model.correct_mu4 = True
    params.options.append('mu4corr')

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

    
