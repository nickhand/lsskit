"""
 fiber_collision_analysis.py
 a few helper functions for analysis
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 09/04/2014
"""
from . import power_measurement, tools
import plotify as pfy
from glob import glob
import numpy as np

#-------------------------------------------------------------------------------
def plot_collision_comparison(files_with, files_without, output_units,
                                data_type="Pkmu",
                                bias_with=1., 
                                bias_without=1., 
                                mu_avg=False, 
                                mu=None, 
                                power_kwargs={'transfer_fit' : 'EH'},
                                subtract_shot_noise=True):
    """
    This will plot P(k, mu) for two samples, averaging over several 
    simulation boxes
    """
    
    labels = ['with fiber collisions', 'without fiber collisions']
    biases = [bias_with, bias_without]
    for i, files in enumerate([files_with, files_without]):
        
        # load the first one for shot noise, etc
        Pkmu0 = power_measurement.load(files[0])
        Pkmu0.output_units = output_units
    
        Pshot = 0.
        if subtract_shot_noise: Pshot = Pkmu0.Pshot

        # first, plot with fiber collisions
        Pk = power_measurement.average_power(files, output_units=output_units, mu_avg=mu_avg, mu=mu, data_type=data_type)
        norm = Pkmu0.normalization(biases[i], power_kwargs, mu_avg=mu_avg, mu=mu, data_type=data_type)
        pfy.errorbar(Pkmu0.ks, (Pk.power-Pshot)/norm, Pk.variance**0.5/norm, label=labels[i])
#end plot_collision_comparison

#-------------------------------------------------------------------------------   
def plot_fractional_power_ratio(files_A, files_B, output_units,
                                data_type="pkmu",
                                bias_A=1., 
                                bias_B=1.,
                                mu_avg=False, 
                                power_kwargs={'transfer_fit' : 'EH'},
                                subtract_shot_noise=True,
                                normalize=True, 
                                **kwargs):
    """
    Plots the fractional power ratio `(P_A - P_B) / |P_B|` as a function of k, 
    either for a series of mu values or the mu-averaged result.
    """
    # compute the means first
    power_A = tools.weighted_mean([power_measurement.load(f)[data_type] for f in files_A])
    power_B = tools.weighted_mean([power_measurement.load(f)[data_type] for f in files_B])

    # update the values
    power_A.update(bias=bias, normalize=normalize, output_units=output_units, subtract_shot_noise=subtract_shot_noise)
    power_B.update(bias=bias, normalize=normalize, output_units=output_units, subtract_shot_noise=subtract_shot_noise)

    # the ratio
    ratio = (power_A - power_B)/abs(power_B)
    
    if mu_avg:
        x = ratio.mu_averaged()
        pfy.errorbar(power_A.ks, x.power, x.variance**0.5, label=r'$\mu$-averaged', **kwargs)  
    else:
        

    ax = pfy.gca()
    ax.axhline(y=0, c='k', lw=1., ls='--')        
    
    # make it look nice
    if output_units == 'relative':
        ax.xlabel.update(r"$\mathrm{k \ (h/Mpc)}$", fontsize=15)
    else:
        ax.xlabel.update(r"$\mathrm{k \ (1/Mpc)}$", fontsize=15)
    ax.ylabel.update(r"$\mathrm{(P_{with} - P_{without}) / |P_{without}|}$", fontsize=15)

#end plot_power_ration

#-------------------------------------------------------------------------------

