"""
 pkmu_measurement.py
 class to hold a P(k, mu) measurement
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 07/18/2014
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
def load(filename):
    """
    Load a ``PkmuMeasurement`` object from a HDF5 file using the pandas module
    
    Parameters
    ----------
    filename : str
        The name of the pickled file
    
    Returns
    -------
    pkmu : PkmuMeasurement object
        The PkmuMeasurement class holding the pandas.DataFrame as `self.data`
    """
    store = pd.HDFStore(filename, 'r')
    
    pkmu = store.get_storer('data').attrs.Pkmu
    pkmu.data = store['data']
    store.close()
    
    return pkmu
#end load

#-------------------------------------------------------------------------------
class PkmuMeasurement(object):
    """
    Class to hold a P(k, mu) measurement, which is essentially a wrapper around
    a ``pandas.DataFrame`` object which holds the P(k, mu) data
    """
    def __init__(self, data_frame, cosmo=None, **kwargs):
        """
        Initialize with the ``pandas.DataFrame`` holding the P(k, mu) data
        
        Parameters
        ----------
        data_frame : ``pandas.DataFrame``
            `DataFrame` holding the P(k, mu) measurement. The `DataFrame` 
            should have a `MultiIndex` with levels `[`mu`, `k`]`, which defines
            each measurement. The columns for the `DataFrame` are:
                `power` : the comoving power measurement with units specified
                          by `self.units`
                `error` : the error on the power measurement
                `baseline` : the baseline power measurement that the power 
                             is measured with respect to
                `noise` : the component of the baseline that is `noise-like`
        cosmo : `cosmology.Cosmology`
            The cosmology with which the power measurement was made 
        """
        self.data = data_frame
        self.cosmo = cosmo
        
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
        
    #end __init__
    
    #---------------------------------------------------------------------------
    @property
    def _mu_level(self):
        """
        Internal variable defining which level the `mu` column is defined in 
        the `MultiIndex`
        """
        try:
            return self.__mu_level
        except AttributeError:
            if "mu" not in self.data.index.names:
                raise ValueError("MultiIndex for P(k, mu) frame does not contain 'mu'")
            index = np.where(np.asarray(self.data.index.names) == 'mu')[0]
            self.__mu_level = index[0]
            return self.__mu_level
    #end _mu_level
    
    #---------------------------------------------------------------------------
    @property
    def _k_level(self):
        """
        Internal variable defining which level the `k` column is defined in 
        the `MultiIndex`
        """
        try:
            return self.__k_level
        except AttributeError:
            if "k" not in self.data.index.names:
                raise ValueError("MultiIndex for P(k, mu) frame does not contain 'k'")
            index = np.where(np.asarray(self.data.index.names) == 'k')[0]
            self.__k_level = index[0]
            return self.__k_level
    #end _k_level
    
    #---------------------------------------------------------------------------
    @property
    def columns(self):
        """
        Return the columns of the ``pandas.DataFrame`` storing the P(k, mu)
        measurement
        
        Returns
        -------
        columns : list
             A list of string column names in the `DataFrame` 
        """
        return list(self.data.columns)
    #end columns
        
    #---------------------------------------------------------------------------
    @property
    def mus(self):
        """
        The mu values where the measurement is defined.
        
        Returns
        -------
        mus : np.ndarray
            An array holding the mu values at which the power is measured
        """
        return np.asarray(self.data.index.levels[self._mu_level], dtype=float)
    #end mus
    
    #---------------------------------------------------------------------------
    @property
    def ks(self):
        """
        The k values where the measurement is defined
        
        Returns
        -------
        ks : np.ndarray
            An array holding the k values at which the power is measured
        """
        return np.ndarray(self.data.index.levels[self._k_level], dtype=float)
    #end ks
    
    #---------------------------------------------------------------------------
    def mu_averaged(self, weighted=True):
        """
        Return the mu averaged power measurement, as a ``pandas.DataFrame``
        object, optionally computing the weighted average of the power for a 
        given ``mu`` value.
        
        Parameters
        ----------
        weighted : bool, optional
            If `True`, compute the weighted average, using the `error` column
            as weights to average the `power` column over `mu`. Default is 
            `True`
        
        Returns
        -------
        avg_data : pandas.DataFrame
            A DataFrame with index corresponding to `self.ks`, holding the 
            power averaged over `mu`
        """
        grouped = self.data.groupby(level=['k'])
        avg_data = grouped.mean()
        
        if weighted:
            power_average = lambda row: np.average(row['power'], weights=row['error']**(-2))
            err_average = lambda row: np.mean(1./row['error']**(-2.))
        
            avg_data['power'] = grouped.apply(power_average)
            avg_data['error'] = np.sqrt(grouped.apply(power_average))
            
        return avg_data
    #end mu_averaged
    
    #---------------------------------------------------------------------------
    def Pk(self, mu):
        """
        Return the power measured P(k) at a specific value of mu
        
        Parameters
        ---------
        mu : int, float
            If a `float`, `mu` must be a value in `self.mus`, or if an `int`, the
            value of `mu` used will be `self.mus[mu]`
        """
        # get the correct value of mu, if we specified an int
        if isinstance(mu, int):
            mu = self.mus[mu]
            
        # make sure the power is defined at this value of mu
        if mu not in self.mus:
            raise ValueError("Power measurement not defined at mu = %s" %mu)
        
        return self.data.xs(mu, level=self._mu_level)
    #end Pk
    
    #---------------------------------------------------------------------------
    def Pmu(self, k):
        """
        Return the power measured P(mu) at a specific value of k
        
        Parameters
        ---------
        k : int, float
            If a `float`, `k` must be a value in `self.ks`, or if an `int`, the
            value of `k` used will be `self.ks[k]`
        """
        # get the correct value of k, if we specified an int
        if isinstance(k, int):
            k = self.ks[k]
            
        # make sure the power is defined at this value of k
        if k not in self.ks:
            raise ValueError("Power measurement not defined at k = %s" %k)
        
        return self.data.xs(k, level=self._k_level)
    #end Pmu
    
    #---------------------------------------------------------------------------
    def save(self, filename):
        """
        Save the `PkmuMeasurement` instance as a pickle to the filename specified
        
        Parameters
        ----------
        filename : str 
            the filename to output to
        """        
        # first write out the galaxies
        store = pd.HDFStore(filename, 'w')
        
        # store the data DataFrame
        store['data'] = self.data
        
        # now also store self
        self.data = None
        store.get_storer('data').attrs.Pkmu = self
        store.close()
    #end save   
    
    #---------------------------------------------------------------------------
    def plot_Pk(self, mu, bias=1., norm_linear=True, plot_linear=False, 
                subtract_shot_noise=False, power_kwargs={}, 
                plot_kwargs={}, curr_ax=None, label=None):
        """
        Plot the measurement

        Parameters
        ----------
        mu : int, float
            If a `float`, `mu` must be a value in `self.mus`, or if an `int`, the
            value of `mu` used will be `self.mus[mu]`
        bias : {float, str} optional
            If a `float`, the bias of the sample. If a `str`, the name of file
            that contains the best-fit linear bias value. Default is `1`
        norm_linear : bool, optional
            Plot the power spectrum normalized by the linear spectrum. Default
            is `True`
        plot_linear : bool, optional
            Plot the linear spectrum as well as the data spectrum. Default
            is `False`
        power_kwargs : dict, optional
            Any keyword arguments to pass to the linear power spectrum `Power`
            class
        subtract_shot_noise : bool, optional
            If `True`, subtract the shot noise
        plot_kwargs : dict, optional
            Any plotting keywords to use.
        curr_ax : Axes, optional
            The current axes to plot on
        """
       from cosmology.growth import Power, growth_rate
       from scipy.interpolate import InterpolatedUnivariateSpline as spline

       default = {'ls':'', 'c':'DodgerBlue', 'marker': '.'}
       default.update(plot_kwargs)

       if curr_ax is None:
           curr_ax = plt.gca()

       # now read in the k, Pk, err data
       k = np.asarray(list(data.index))
       Pk = np.asarray(data.power)
       err = np.asarray(data.error)

       # put into right units
       if 'h' not in header['units'] and self.units == 'Mpc/h':
           k /= self.cosmo.h
           Pk *= self.cosmo.h**3
           err *= self.cosmo.h**3 

       Pshot = 0.
       if subtract_shot_noise:

           volume = self.box_size**3
           if 'h' not in self.units:
               volume *= self.cosmo.h**3
           Pshot = (volume / header['sample_size'])

       # plot the linear theory result
       if plot_linear or norm_linear:

           k_th = np.logspace(np.log10(np.amin(k)), np.log10(np.amax(k)), 1000)
           power = Power(k=k_th, z=self.redshift, cosmo=self.cosmo, **power_kwargs)

           if header['space'] == 'real':
               Plin_norm = bias**2 * power.power
           else:
               f = growth_rate(self.redshift, params=self.cosmo)
               Plin_norm = (bias + f*mu**2)**2 * power.power

           if plot_linear:

               # plot both the linear and result separately
               curr_ax.loglog(power.k, Plin_norm, c='k', label=r'linear $P(k, \mu)$')
               curr_ax.errorbar(k, Pk-Pshot, err, label=r"$P(k, \mu = %.3f)$" %mu, **default)
           else:
               norm_spline = spline(power.k, Plin_norm)
               Plin_norm = norm_spline(k)

               # plot the normalized result
               curr_ax.errorbar(k, (Pk - Pshot)/Plin_norm, err/Plin_norm, 
                                   label=r"$P(k, \mu = %.3f)$" %mu, **default)
       else:
           curr_ax.errorbar(k, (Pk - Pshot), err, label=r"$P(k, \mu = %.3f)$" %mu, **default)
           curr_ax.set_xscale('log')
           curr_ax.set_yscale('log')

       if self.units == 'Mpc':
           k_units = '1/Mpc'
       else:
           k_units = 'h/Mpc'
       curr_ax.set_xlabel(r"$\mathrm{k \ (%s)}$" %k_units, fontsize=18)

       if subtract_shot_noise:
           s = r"P(k, \mu) - \bar{n}^{-1}"
       else:
           s = r"P(k, \mu)"

       if norm_linear:
           curr_ax.set_ylabel(r"$\mathrm{(%s) \ / \ P_{NW}(k, \mu)}$" %s, fontsize=18)
       else:    
           curr_ax.set_ylabel(r"$\mathrm{%s %s}$" %(s, header['units']), fontsize=18)

       return curr_ax
    #end plot_Pkmu    
        
        