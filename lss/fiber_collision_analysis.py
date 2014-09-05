"""
 fiber_collision_analysis.py
 a few helper functions for analysis
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 09/04/2014
"""
from . import pkmu_measurement
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
        Pkmu0 = pkmu_measurement.load(files[0])
        Pkmu0.output_units = output_units
    
        Pshot = 0.
        if subtract_shot_noise: Pshot = Pkmu0.Pshot

        # first, plot with fiber collisions
        Pk = pkmu_measurement.average_power(files, output_units=output_units, mu_avg=mu_avg, mu=mu, data_type=data_type)
        norm = Pkmu0.normalization(biases[i], power_kwargs, mu_avg=mu_avg, mu=mu, data_type=data_type)
        pfy.errorbar(Pkmu0.ks, (Pk.power-Pshot)/norm, Pk.error/norm, label=labels[i])
#end plot_collision_comparison

#-------------------------------------------------------------------------------   
def plot_power_ratio(files_with, files_without, output_units,
                        data_type="Pkmu",
                        bias_with=1., 
                        bias_without=1.,
                        mu_avg=False, 
                        power_kwargs={'transfer_fit' : 'EH'},
                        subtract_shot_noise=True):
    """
    Plots the ratio Pk_with / Pk_without as a function of k, either for a series
    of mu values or the mu-averaged result. Pk_with is the **normalized** 
    power spectrum for the sample with fiber collisions
    """
    
    # load the first measurement
    Pkmu0_with = pkmu_measurement.load(files_with[0])
    Pkmu0_without = pkmu_measurement.load(files_without[0])
    
    # set the output units
    Pkmu0_with.output_units = output_units
    Pkmu0_without.output_units = output_units

    # shot noises
    Pshot_with = 0.
    if subtract_shot_noise: Pshot_with = Pkmu0_with.Pshot
    
    Pshot_without = 0.
    if subtract_shot_noise: Pshot_without = Pkmu0_without.Pshot
    
    # do a mu-average
    if mu_avg:
        
        if data_type != "Pkmu":
            raise ValueError("Confused: cannot do mu-avg when data_type != 'Pkmu'")
        
        # get the weighted frames
        avg_with = pkmu_measurement.average_power(files_with, output_units=output_units, mu_avg=True, data_type="Pkmu")
        avg_without = pkmu_measurement.average_power(files_without, output_units=output_units, mu_avg=True, data_type="Pkmu")
                
        # these are the normalizations
        norm_with = Pkmu0_with.normalization(bias_with, power_kwargs, mu_avg=True, data_type=data_type)
        norm_without = Pkmu0_without.normalization(bias_without, power_kwargs, mu_avg=True, data_type=data_type)
        
        # now get the normalized results
        a = (avg_without.power - Pshot_without) / norm_without
        b = (avg_with.power - Pshot_with) / norm_with
        
        a_err = avg_without.error / norm_without
        b_err = avg_with.error / norm_with
        delta  = b  - a  
        delta_err = np.sqrt(a_err**2 + b_err**2)
        
        diff = delta / abs(a)
        diff_err = diff * np.sqrt( (delta_err/delta)**2 + (a_err/a)**2 )
        pfy.errorbar(Pkmu0_with.ks, diff, diff_err, ls='-',  color='k', linewidth=1.5, zorder=10, label=r"$\mu$-avg")

    else:

        if data_type == "Pkmu":
            
            # loop over each mu
            for mu in Pkmu0_with.mus:

                # get the weighted frames
                Pk_with = pkmu_measurement.average_power(files_with, output_units=output_units, mu=mu, data_type="Pkmu")
                Pk_without = pkmu_measurement.average_power(files_without, output_units=output_units, mu=mu, data_type="Pkmu")
 
                # these are the normalizations
                norm_with = Pkmu0_with.normalization(bias_with, power_kwargs, mu=mu, mu_avg=False, data_type="Pkmu")
                norm_without = Pkmu0_without.normalization(bias_without, power_kwargs, mu=mu, mu_avg=False, data_type="Pkmu")

                
                # now get the normalized results
                a = (Pk_without.power - Pshot_without) / norm_without
                b = (Pk_with.power - Pshot_with) / norm_with
            
                a_err = Pk_without.error / norm_without
                b_err = Pk_with.error / norm_with
                delta  = b - a
                delta_err = np.sqrt(a_err**2 + b_err**2)
        
                diff = delta / abs(a)
                diff_err = diff * np.sqrt( (delta_err/delta)**2 + (a_err/a)**2 )

                pfy.errorbar(Pkmu0_with.ks, diff, diff_err, ls='-', linewidth=1.0, label=r"$\mu$ = %.1f" %mu)
        else:
            # get the weighted frames
            Pk_with = pkmu_measurement.average_power(files_with, output_units=output_units, data_type=data_type)
            Pk_without = pkmu_measurement.average_power(files_without, output_units=output_units, data_type=data_type)

            # these are the normalizations
            norm_with = Pkmu0_with.normalization(bias_with, power_kwargs, data_type=data_type)
            norm_without = Pkmu0_without.normalization(bias_without, power_kwargs, data_type=data_type)

            # now get the normalized results
            a = (Pk_without.power - Pshot_without) / norm_without
            b = (Pk_with.power - Pshot_with) / norm_with

            a_err = Pk_without.error / norm_without
            b_err = Pk_with.error / norm_with
            delta  = b - a 
            delta_err = np.sqrt(a_err**2 + b_err**2)

            diff = delta / abs(a)
            diff_err = diff * np.sqrt( (delta_err/delta)**2 + (a_err/a)**2 )

            pfy.errorbar(Pkmu0_with.ks, diff, diff_err, ls='-', linewidth=1.0, label=data_type)
        

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

