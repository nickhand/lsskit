"""
 pkmu_measurement.py
 class to hold a P(k, mu) measurement
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 07/18/2014
"""
import numpy as np
from pandas import HDFStore

import plotify as pfy
from . import pkmu_driver
from cosmology.growth import Power, growth_rate

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
    store = HDFStore(filename, 'r')
    
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
                `noise` : the component of the baseline that is "noise-like"
        cosmo : `cosmology.Cosmology`
            The cosmology with which the power measurement was made 
        """
        self.data = data_frame
        self.cosmo = cosmo
        
        # the keywords 
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
            
        # compute the growth rate f for convenience
        self.f = growth_rate(self.redshift, params=self.cosmo)
        
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
        return np.asarray(self.data.index.levels[self._k_level], dtype=float)
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
            avg_data['error'] = np.sqrt(grouped.apply(err_average))
        else:
            avg_data['error'] /= np.sqrt(len(self.mus))
            
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
    def Pk_kaiser(self, bias, power_kwargs):
        """
        Return the biased, Kaiser real-space power spectrum at the `k` 
        values defined for this measurement. This is independent of `mu`.
        """
        if hasattr(self, 'power_kwargs') and self.power_kwargs == power_kwargs:
            return bias**2 * self._Pk_lin
        else:
            self.power_kwargs = power_kwargs
            
            ks = self.ks / self.cosmo.h if 'h' not in self.units else self.ks
            power = Power(k=ks, z=self.redshift, cosmo=self.cosmo, **self.power_kwargs)
            
            self._Pk_lin = power.power
            return bias**2 * self._Pk_lin
            
    #---------------------------------------------------------------------------
    def Pkmu_kaiser(self, mu, bias, power_kwargs):
        """
        Return the biased, Kaiser redshift-space power spectrum at the `k` values 
        defined for this measurement and the input `mu` value. 
        """
        beta = self.f / bias
        return (1. + beta*mu**2)**2 * self.Pk_kaiser(bias, power_kwargs)
            
    #---------------------------------------------------------------------------
    def mu_averaged_Pkmu_kaiser(self, bias, power_kwargs):
        """
        Return the mu-averaged, biased, Kaiser redshift-space power spectrum 
        at the `k` values defined for this measurement and the input `mu` value. 
        """
        norms = []
        for mu in self.mus:
            norms.append(self.Pkmu_kaiser(mu, bias, power_kwargs))
        Plin_norm = np.mean(norms, axis=0)
        return Plin_norm
            
    #---------------------------------------------------------------------------
    def normalization(self, bias, power_kwargs, mu=None, mu_avg=False):
        """
        Return the proper Kaiser normalization for this power measurement
        """
        # determine which normalization to use
        if self.space == 'real':
            Plin_norm = self.Pk_kaiser(bias, power_kwargs)
        elif self.space == 'redshift':
            
            if mu_avg:
                Plin_norm = self.mu_averaged_Pkmu_kaiser(bias, power_kwargs)
            else:
                Plin_norm = self.Pkmu_kaiser(mu, bias, power_kwargs)
        else:
            raise ValueError("Attribute `space` must be one of {`real`, `redshift`}")
        
        return Plin_norm
            
    #---------------------------------------------------------------------------
    @property
    def Pshot(self):
        """
        Return the shot noise, in (Mpc/h)^3
        """
        try:
            return self._Pshot
        except AttributeError:
            if not hasattr(self, 'volume'):
                raise AttributeError("Need to specify the `volume` attribute.")
            if not hasattr(self, 'sample_size'):
                raise AttributeError("Need to specify the `sample_size` attribute.")

            volume = self.volume
            if 'h' not in self.units:
                volume *= self.cosmo.h**3
            self._Pshot = (volume / self.sample_size)
            return self._Pshot
    #---------------------------------------------------------------------------
    def add_multipoles(self, tsal_file):
        """
        Add a monopole and quadrupole measurement, where the multipoles are
        determined by fitting the following model to the P(k, mu) measurement
        
          :math: P(k, mu) = monopole(k) + 0.5*(3 mu^2 - 1)*quadrupole(k)       
        """
        poles = pkmu_driver.extract_multipoles(tsal_file)
        ks, self._mono, self._mono_err, self._quad, self._quad_err = poles
        
        if np.any(self.ks != ks):
            raise ValueError("Wavenumber mismatch between P(k, mu) and multipole measurements")
            
    #end add_multipoles
    
    #---------------------------------------------------------------------------
    @property
    def monopole(self):
        """
        Return the monopole multipole values and errors
        """
        try:
            return self._mono, self._mono_err
        except:
            raise AttributeError("No monopole measurement provided.")

    @property
    def quadrupole(self):
        """
        Return the quadrupole multipole values and errors
        """
        try:
            return self._quad, self._quad_err
        except:
            raise AttributeError("No quadrupole measurement provided.")
    
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
        store = HDFStore(filename, 'w')
        
        # store the data DataFrame
        store['data'] = self.data
        
        # now also store self
        self.data = None
        store.get_storer('data').attrs.Pkmu = self
        store.close()
    #end save   
    
    #---------------------------------------------------------------------------
    def plot_Pk(self, *args, **kwargs):
        """
        Plot the P(k, mu) measurement

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
        ax : plotify.Axes, optional
            This can be specified as the first positional argument, or as the
            keyword `ax`. If not specified, the plot will be added to 
            the axes returned by `plotify.gca()`
        label : str, optional
            The label to attach to this plot lines
        offset : float, optional
            Offset the plot in the y-direction by this amount.
        weighted_mean : bool, optional
            Take the weighted mu-average. Default is `False`
        
        Returns
        -------
        fig : plotify.Figure
            The figure where the lines were plotted
        ax : plotify.Axes
            The axes where the lines were plotted
        Pshot : float
            The shot noise for this sample
        """
        from os.path import exists
        
        # first parse the arguments to see if we have an axis instance
        ax, args, kwargs = pfy.parse_arguments(*args, **kwargs)       
        
        # default keyword values
        weighted_mean = kwargs.get('weighted_mean', False)
        label = kwargs.get('label', None)
        y_offset = kwargs.get('offset', 0.)
        plot_kwargs = kwargs.get('plot_kwargs', {})
        power_kwargs = kwargs.get('power_kwargs', {})
        subtract_shot_noise = kwargs.get('subtract_shot_noise', False)
        bias = kwargs.get('bias', 1.)
        if isinstance(bias, basestring):
            assert exists(bias), "Specified bias TSAL file does not exist"
            bias, bias_err = pkmu_driver.extract_bias(bias)
            
        # determine the keywords for normalization
        plot_linear = kwargs.get('plot_linear', False)
        norm_linear = kwargs.get('norm_linear', False)
        assert not (plot_linear == True and norm_linear == True), \
            "Boolean keywords `norm_linear` and `plot_linear` must take different values."
       
        # let's get the data
        mu_avg = False
        if kwargs.get('mu', None) is None:
            mu_avg = True
            mu   = np.mean(self.mus)
            data = self.mu_averaged(weighted=weighted_mean)
        else:
            # get the mu value, correctly 
            mu = kwargs['mu']
            if isinstance(mu, int):
                mu = self.mus[mu]
            data = self.Pk(mu)
        k    = self.ks
        Pk   = data.power
        err  = data.error
      
        # now let's make sure we have units
        if not hasattr(self, 'units'):
            raise AttributeError("Need to specify the `units` attribute.")
        
        # this goes from (Mpc)^3 to (Mpc/h)^3)
        # this transforms
        if 'h' not in self.units:
            k /= self.cosmo.h      # should be in h/Mpc
            Pk *= self.cosmo.h**3  # should be in (Mpc/h)^3
            err *= self.cosmo.h**3 # should be in (Mpc/h)^3

        # the shot noise
        Pshot_sub = 0.
        if subtract_shot_noise: Pshot_sub = self.Pshot
          
        # plot the linear theory result
        if plot_linear or norm_linear:

            # the normalization
            Plin_norm = self.normalization(bias, power_kwargs, mu=mu, mu_avg=mu_avg)
    
            # plot both the linear and result separately
            if plot_linear:
                if label is None: label = r'linear $P(k, \mu)$'
                pfy.loglog(ax, k, Plin_norm, c='k', label=label)
                
                if label is None: label = r"$P(k, \mu = %.3f)$ "%mu
                pfy.errorbar(ax, k, Pk-Pshot_sub, err, label=label, **plot_kwargs)
            else:
                # plot the normalized result
                data_to_plot = (k, (Pk-Pshot_sub)/Plin_norm, err/Plin_norm)
                pfy.plot_data(ax, data=data_to_plot, labels=label, y_offset=y_offset, plot_kwargs=plot_kwargs)
        else:
            
            if label is None: label = r"$P(k, \mu = %.3f)$" %mu
            pfy.errorbar(ax, k, (Pk - Pshot_sub), err, label=label, **plot_kwargs)
            ax.x_log_scale()
            ax.y_log_scale()

        # let's set the xlabels and return
        if ax.xlabel.text == "":
            ax.xlabel.update(r"$\mathrm{k \ (h/Mpc)}$")
            
        if ax.ylabel.text == "":
            if mu_avg: 
                Pkmu_label = r"\langle P(k, \mu) \rangle"
            else:
                Pkmu_label = r"P(k, \mu)"
            if norm_linear:
                if subtract_shot_noise:
                    ax.ylabel.update(r"$\mathrm{(%s - \bar{n}^{-1}) \ / \ P_\mathrm{nw}(k, \mu)}$" %(Pkmu_label))
                else:
                    ax.ylabel.update(r"$\mathrm{%s \ / \ P_\mathrm{nw}(k, \mu)}$" %(Pkmu_label))
            
            else:    
                ax.ylabel.update(r"$\mathrm{%s \ (Mpc/h)^3}$" %(Pkmu_label))

        return ax.get_figure(), ax
    
    #end plot_Pkmu    
    #---------------------------------------------------------------------------

#endclass PkmuMeasurement
#-------------------------------------------------------------------------------
        
        
