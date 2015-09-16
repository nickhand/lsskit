"""
    base.py
    lsskit.rsdfit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : base fitting parameter set
"""
from . import TheoryParams

#------------------------------------------------------------------------------
# Base model
#------------------------------------------------------------------------------
class BaseTheoryParams(TheoryParams):
    """
    Base fitting parameter set with 14 free parameters
    
    Notes
    -----
    * both 1-halo amplitude params `f1h_cBs` and `f1h_sBsB` are varying
    * no SO corrections, so `f_so` and `sigma_so` are set to 0
    """
    _name = 'base'
    
    def __init__(self):
        super(BaseTheoryParams, self).__init__()
        
        # cosmo
        self.sigma8_z.vary   = True
        self.f.vary          = True
        self.alpha_perp.vary = True
        self.alpha_par.vary  = True

        # biases
        self.b1_cA.vary = True
        self.b1_cB.update(vary=False, expr="(1-fsB**2)*b1_sA + fsB**2*b1_sB")
        self.b1_sA.update(vary=False, expr="gamma_b1sA*b1_cA")
        self.b1_sB.update(vary=False, expr="gamma_b1sB*b1_cA")
    
        # fractions
        self.fcB.update(vary=False, expr="fs / (1 - fs) * (1 + fsB*(1./Nsat_mult - 1))")
        self.fsB.vary = True
        self.fs.vary = True
        self.Nsat_mult.vary = True
        self.f_so.update(vary=False, fiducial=0.)

        # sigmas
        self.sigma_c.vary = True
        self.sigma_s.vary = True
        self.sigma_sA.update(vary=False, expr="sigma_s * sigmav_from_bias(b1_sA) / sigmav_from_bias(b1_s)")
        self.sigma_sB.update(vary=False, expr="sigma_s * sigmav_from_bias(b1_sB) / sigmav_from_bias(b1_s)")
        self.sigma_so.update(vary=False, fiducial=0.)

        # amplitudes
        self.NcBs.update(vary=False, expr="f1h_cBs / (fcB*(1 - fs)*nbar)")
        self.NsBsB.update(vary=False, expr="f1h_sBsB / (fsB**2 * fs**2 * nbar) * (fcB*(1 - fs) - fs*(1-fsB))")
        
        # nuisance
        self.gamma_b1sA.vary = True
        self.gamma_b1sB.vary = True
        self.f1h_sBsB.vary = True
        self.f1h_cBs.vary = True   
 
    
