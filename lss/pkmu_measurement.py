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
from . import tools

from cosmology.growth import Power, growth_rate
from cosmology.parameters import Cosmology
from cosmology.utils.units import h_conversion_factor

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
    def __init__(self, data_frame, units, **kwargs):
        """
        Initialize with the ``pandas.DataFrame`` holding the P(k, mu) data
        
        Parameters
        ----------
        data_frame : ``pandas.DataFrame``
            `DataFrame` holding the P(k, mu) measurement. The `DataFrame` 
            should have a `MultiIndex` with levels `[`mu`, `k`]`, which defines
            each measurement. The columns for the `DataFrame` are:
                `power` : the comoving power measurement with units specified
                          by `units`
                `error` : the error on the power measurement
                `baseline` : the baseline power measurement that the power 
                             is measured with respect to
                `noise` : the component of the baseline that is "noise-like"
        
        units : str, {`absolute`, `relative`}
            The units of the power spectrum measurement. `Absolute` means
            the power is measured in `Mpc^3`, while `relative` means the 
            power is measured relative to the dimensionless hubble constant, 
            in units of `(Mpc/h)^3`
            
        kwargs : dict
            Additional keyword arguments to store as attributes. Some useful
            keywords are:
            
            cosmo : {`cosmology.parameters.Cosmology`, dict}
                A dict-like object holding the cosmology parameterws with which the 
                power measurement was made. For unit conversions, `h` must be 
                defined. Default is `None`.
        
            redshift : float
                The redshift at which the measurement was made. Default is `None`.
                
            space : str, {`real`, `redshift`, `None`}
                Whether the measurement was made in real or redshift space. 
                Default is `None`
        """
        # the data frame storing the P(k, mu) data
        self.data = data_frame
        
        # dictionary storing the cosmology
        self.cosmo = kwargs.pop('cosmo', None)
        if self.cosmo is not None:
            if not isinstance(self.cosmo, (dict, Cosmology)):
                raise TypeError("The cosmology object must be one of [dict, Cosmology]")
            
        # the redshift of the measurement
        self.redshift = kwargs.pop('redshift', None)
        
        # the space of the measurement
        self.space = kwargs.pop('space', None)
        
        # the measurement units
        if units not in ['absolute', 'relative']:
            raise ValueError("`Units` must be one of [`absolute`, `relative`]")
        self.measurement_units = units
        
        # any other keywords
        for k, v in kwargs.iteritems():
            setattr(self, k, v)
            
        # compute the growth rate f for convenience, if we have a redshift
        # and a cosmology
        if self.cosmo is not None and self.redshift is not None:
            self.f = growth_rate(self.redshift, params=self.cosmo)
            
        # keep track of the units for output
        self._output_units = self.measurement_units
        
    #end __init__
    #---------------------------------------------------------------------------
    @property
    def output_units(self):
        """
        The type of units to use for output quanities. Either `relative` or 
        `absolute`. Initialized to `self.measurement_units`
        """
        return self._output_units
        
    @output_units.setter
    def output_units(self, val):
        
        if val not in ['absolute', 'relative']:
            raise ValueError("`output_units` must be one of [`absolute`, `relative`]")
        
        if val != self.measurement_units and self.cosmo is None:
            raise ValueError("Cannot convert units of output quantities with a cosmology defined")
            
        self._output_units = val
        
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
    # some additional properties
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
        The k values where the measurement is defined, in units specified
        by `self.output_units`
        
        Returns
        -------
        ks : np.ndarray
            An array holding the k values at which the power is measured
        """
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('wavenumber', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])
                                                        
        return np.asarray(self.data.index.levels[self._k_level], dtype=float)*output_units_factor
    #end ks
    #---------------------------------------------------------------------------
    @property
    def Pshot(self):
        """
        Return the shot noise, computed as volume / sample_size, in units 
        specified by `self.output_units`
        """
        # must have both a volume attribute and a sample_size attribute
        if not hasattr(self, 'volume'):
            raise AttributeError("Need to specify the `volume` attribute.")
        if not hasattr(self, 'sample_size'):
            raise AttributeError("Need to specify the `sample_size` attribute.")

        # conversion factor for output units, as specified in `self.output_units`
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('volume', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])

        return (self.volume / self.sample_size) * output_units_factor
    
    #end Pshot
    
    #---------------------------------------------------------------------------
    @property
    def monopole(self):
        """
        Return a tuple of monopole values and errors, in units specified by 
        `self.output_units`
        
        Returns
        -------
        monopole : np.ndarray
            The measured monopole values in units specified by `self.output_units`,
            and defined at the `k` values in `self.ks`
        monopole_error : np.ndarray
            The measured monopole errors in units specified by `self.output_units`,
            and defined at the `k` values in `self.ks`
        """
        # conversion factor for output units, as specified in `self.output_units`
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('power', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])
        try:
            return self._mono*output_units_factor, self._mono_err*output_units_factor
        except:
            raise AttributeError("No monopole measurement provided yet.")
    
    #end monopole
    
    #---------------------------------------------------------------------------
    @property
    def quadrupole(self):
        """
        Return a tuple of quadrupole values and errors, in units specified by 
        `self.output_units`

        Returns
        -------
        quadrupole : np.ndarray
            The measured quadrupole values in units specified by `self.output_units`,
            and defined at the `k` values in `self.ks`
        quadrupole_error : np.ndarray
            The measured quadrupole errors in units specified by `self.output_units`,
            and defined at the `k` values in `self.ks`
        """
        # conversion factor for output units, as specified in `self.output_units`
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('power', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])
        try:
            return self._quad*output_units_factor, self._quad_err*output_units_factor
        except:
            raise AttributeError("No quadrupole measurement provided yet.")

    #end quadrupole
    
    #---------------------------------------------------------------------------
    # the functions
    #---------------------------------------------------------------------------
    def add_multipoles(self, tsal_file):
        """
        Add a monopole and quadrupole measurement, where the multipoles are
        determined by fitting the following model to the P(k, mu) measurement
        
          :math: P(k, mu) = monopole(k) + 0.5*(3 mu^2 - 1)*quadrupole(k)       
        """
        poles = tools.extract_multipoles(tsal_file)
        ks, self._mono, self._mono_err, self._quad, self._quad_err = poles
        
        if np.any(self.ks != ks):
            raise ValueError("Wavenumber mismatch between P(k, mu) and multipole measurements")
            
    #end add_multipoles
    
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
    def mu_averaged(self, weighted=True):
        """
        Return the mu averaged power measurement, as a ``pandas.DataFrame``
        object, optionally computing the weighted average of the power for a 
        given ``mu`` value. The units of the output are specified by 
        `self.output_units`
        
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
        # units of self.output_units
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('power', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])
        grouped = self.data.groupby(level=['k'])
        avg_data = grouped.mean()
        
        if weighted:
            power_average = lambda row: np.average(row['power'], weights=row['error']**(-2))
            err_average = lambda row: np.mean(1./row['error']**(-2.))
        
            avg_data['power'] = grouped.apply(power_average)
            avg_data['error'] = np.sqrt(grouped.apply(err_average))
        else:
            avg_data['error'] /= np.sqrt(len(self.mus))
            
        return avg_data*output_units_factor
    #end mu_averaged
    
    #---------------------------------------------------------------------------
    def Pk(self, mu):
        """
        Return the power measured P(k) at a specific value of mu, as a 
        `pandas.DataFrame`.  The units of the output are specified by 
        `self.output_units`
        
        Parameters
        ---------
        mu : {int, float}
            If a `float`, `mu` must be a value in `self.mus`, or if an `int`, 
            the value of `mu` used will be `self.mus[mu]`
            
        Returns
        -------
        data : pandas.DataFrame
            The slice of the full DataFrame holding the columns values for
            this input `mu` value
        """
        # units of self.output_units
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('power', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])
                                                        
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
        Return the power measured P(mu) at a specific value of k.  The units of 
        the output are specified by `self.output_units`
        
        Parameters
        ---------
        k : int, float
            If a `float`, `k` must be a value in `self.ks`, or if an `int`, the
            value of `k` used will be `self.ks[k]`
        
        Returns
        -------
        data : pandas.DataFrame
            The slice of the full DataFrame holding the columns values for
            this input `k` value
        """
        # units of self.output_units
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('power', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])
                                                        
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
        The returned expression is 
        
        :math: P_{kaiser}(k) = bias^2 * P_{linear}(k)
        
        Parameters
        ----------
        bias : float
            The linear bias factor
        power_kwargs : dict
            Dictionary of keywords to pass to the `Power` instance, which computes
            the linear power spectrum
            
        Returns
        -------
        Pk_kaiser : np.ndarray
            The real-space Kaiser P(k) in units specified by `self.output_units`,
            and defined at the `k` values of `self.ks`
        """
        if self.cosmo is None or self.redshift is None:
            raise ValueError("Redshift and cosmology must be defined for this quantity.")
            
        # conversion factor for output units, as specified in `self.output_units`
        output_units_factor = h_conversion_factor('power', 'relative', self.output_units, self.cosmo['h'])
        
        # try not to recompute the linear power if nothing changed
        if hasattr(self, '_power_kwargs') and (self._power_kwargs == power_kwargs):
            return (bias**2 * self._Pk_lin) * output_units_factor
        else:
            # save power kwargs for later 
            self._power_kwargs = power_kwargs
            
            # linear power spectrum is in "relative" units, so convert wavenumber
            # units appropriately
            ks = self.ks * h_conversion_factor('wavenumber', self.measurement_units, 'relative', self.cosmo['h'])
            power = Power(k=ks, z=self.redshift, cosmo=self.cosmo, **self._power_kwargs)
            self._Pk_lin = power.power
            
            # return the biased power with correct units
            return (bias**2 * self._Pk_lin) * output_units_factor 
    #end Pk_kaiser
            
    #---------------------------------------------------------------------------
    def Pkmu_kaiser(self, mu, bias, power_kwargs):
        """
        Return the biased, Kaiser redshift-space power spectrum at the `k` values 
        defined for this measurement and the input `mu` value. The returned 
        expression is 
        
        :math: P_{kaiser}(k, \mu) = (bias + f * \mu^2)^2 * P_{linear}(k)
        
        Parameters
        ----------
        mu : float
            The mu value to use.
        bias : float
            The linear bias factor
        power_kwargs : dict
            Dictionary of keywords to pass to the `Power` instance, which computes
            the linear power spectrum
            
        Returns
        -------
        Pkmu_kaiser : np.ndarray
            The redshift-space Kaiser P(k, mu) in units specified by 
            `self.output_units` and defined at the `k` values of `self.ks`
        """
        beta = self.f / bias
        return (1. + beta*mu**2)**2 * self.Pk_kaiser(bias, power_kwargs)
    
    #end Pkmu_kaiser
            
    #---------------------------------------------------------------------------
    def mu_averaged_Pkmu_kaiser(self, bias, power_kwargs):
        """
        Return the mu-averaged, biased, Kaiser redshift-space power spectrum 
        at the `k` values defined for this measurement. This function 
        takes the average of the output of `self.Pkmu_kaiser` for each `mu`
        value defined in `self.mus` 
        
        Parameters
        ----------
        bias : float
            The linear bias factor
        power_kwargs : dict
            Dictionary of keywords to pass to the `Power` instance, which computes
            the linear power spectrum
            
        Returns
        -------
        Pk : np.ndarray
            The mu-averaged, Kaiser redshift space spectrum in units specified 
            by `self.output_units`, and defined at the `k` values of `self.ks`
        """
        return np.mean([self.Pkmu_kaiser(mu, bias, power_kwargs) for mu in self.mus], axis=0)

    #end mu_averaged_Pkmu_kaiser
    
    #---------------------------------------------------------------------------
    def normalization(self, bias, power_kwargs, mu=None, mu_avg=False, data_type="Pkmu"):
        """
        Return the proper Kaiser normalization for this power measurement, 
        mostly used for plotting purposes.
        
        Parameters
        ----------
        bias : float
            The linear bias factor
        power_kwargs : dict
            Dictionary of keywords to pass to the `Power` instance, which computes
            the linear power spectrum
        mu : float, optional
            The mu value to use. This is not used if ``self.space == real``
        mu_avg : bool, optional
            Whether to compute the mu-averaged normalization. Default is `False`
        data_type : str, {`Pkmu`, `monopole`, `quadrupole`}
            The type of data measurement we want to normalize against. If not
            equal to `Pkmu`, the Kaiser monopole is returned
        """
        if data_type != "Pkmu":
            if self.space == 'real':
                return  self.Pk_kaiser(bias, power_kwargs) 
            else:
                
                beta = self.f / bias
                if data_type == 'monopole':
                    return (1. + 2./3.*beta + 1./5.*beta**2) * self.Pk_kaiser(bias, power_kwargs) 
                else:
                    return (4./3.*beta + 4./7.*beta**2) * self.Pk_kaiser(bias, power_kwargs)
        else:
            
            if self.space == 'real':
                Plin_norm = self.Pk_kaiser(bias, power_kwargs)
            elif self.space == 'redshift':
            
                if mu_avg:
                    Plin_norm = self.mu_averaged_Pkmu_kaiser(bias, power_kwargs)
                else:
                    Plin_norm = self.Pkmu_kaiser(mu, bias, power_kwargs)
            else:
                raise ValueError("Attribute `space` must be defined for this function.")
        
        return Plin_norm
        
    #end normalization
            
    #---------------------------------------------------------------------------
    def plot(self, *args, **kwargs):
        """
        Plot either the P(k, mu), monopole, or quadrupole measurement, in
        the units specified by `self.output_units`

        Parameters
        ----------
        ax : plotify.Axes, optional
            This can be specified as the first positional argument, or as the
            keyword `ax`. If not specified, the plot will be added to the 
            axes returned by `plotify.gca()`
            
        type : str, {`Pkmu`, `monopole`, `quadrupole`}
            Positional argument specifying which type of measurment to plot.
            
        mu : {int, float}, optional
            If a `float`, `mu` must be a value in `self.mus`, or if an `int`, the
            value of `mu` used will be `self.mus[mu]`. If not specified
            and `type == Pkmu`, the mu-averaged spectrum will be plotted. Default
            is `None`.
            
        bias : {float, str}, optional
            If a `float`, the bias of the sample. If a `str`, the name of file
            that contains the best-fit linear bias value. Default is `1`.
            
        norm_linear : bool, optional
            Plot the power spectrum normalized by the linear spectrum. Default
            is `False`.
            
        plot_linear : bool, optional
            Plot the linear spectrum as well as the data spectrum. Default
            is `False`.
            
        power_kwargs : dict, optional
            Any keyword arguments to pass to the linear power spectrum `Power`
            class. Default is `{}`.
            
        subtract_shot_noise : bool, optional
            If `True`, subtract the shot noise before plotting. Default 
            is `False`.
            
        plot_kwargs : dict, optional
            Any additional keywords to use when plotting. Default is `{}`.
        
        label : str, optional
            The label to attach to these plot lines. Default is `None`
        
        offset : float, optional
            Offset the plot in the y-direction by this amount. Default is `0`
        
        weighted_mean : bool, optional
            When averaging the power spectrum over `mu`, whether or not
            to use a weighted average. Default is `False`
        
        Returns
        -------
        fig : plotify.Figure
            The figure where the lines were plotted
        ax : plotify.Axes
            The axes where the lines were plotted
        """
        from os.path import exists
        
        # first parse the arguments to see if we have an axis instance
        ax, args, kwargs = pfy.parse_arguments(*args, **kwargs)       
        
        # default keyword values
        bias                = kwargs.get('bias', 1.)
        norm_linear         = kwargs.get('norm_linear', False)
        plot_linear         = kwargs.get('plot_linear', False)
        power_kwargs        = kwargs.get('power_kwargs', {})
        subtract_shot_noise = kwargs.get('subtract_shot_noise', False)
        plot_kwargs         = kwargs.get('plot_kwargs', {})
        label               = kwargs.get('label', None)
        y_offset            = kwargs.get('offset', 0.)
        weighted_mean       = kwargs.get('weighted_mean', False)

        # check if we need to extract a bias value
        if isinstance(bias, basestring):
            assert exists(bias), "Specified bias TSAL file does not exist"
            bias, bias_err = tools.extract_bias(bias)
            
        # check that both plot_linear and norm_linear are not True
        assert not (plot_linear == True and norm_linear == True), \
            "Boolean keywords `norm_linear` and `plot_linear` must take different values."
       
        # should be one argument now
        if len(args) != 1:
            raise ValueError("Do not understand the input positional arguments")
        
        # check the value of type
        data_type = args[0]
        if data_type not in ['Pkmu', 'monopole', 'quadrupole']:
            raise ValueError("Must specify a `type` positional argument and it must be one of ['Pkmu', 'monopole', 'quadrupole']")
           
        # now get the data
        k = self.ks 
        if data_type == 'Pkmu':
            # mu-averaged or not
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
            Pk   = data.power
            err  = data.error
        else:
            mu = mu_avg = None
            if data_type == 'monopole':
                Pk, err = self.monopole
            else:
                Pk, err = self.quadrupole
              
        # the shot noise
        Pshot_sub = 0.
        if subtract_shot_noise: Pshot_sub = self.Pshot
          
        # plot the linear theory result
        if plot_linear or norm_linear:

            # the normalization
            if norm_linear and data_type == 'quadrupole':
                norm = self.normalization(bias, power_kwargs, mu=mu, mu_avg=mu_avg, data_type='monopole')
            else:
                norm = self.normalization(bias, power_kwargs, mu=mu, mu_avg=mu_avg, data_type=data_type)
    
            # plot both the linear normalization and result separately
            if plot_linear:
                
                # set up labels and plot the linear normalization
                if data_type == "Pkmu":
                    lin_label = r'$P^\mathrm{EH}(k, \mu)$'
                    P_label = r"$P(k, \mu = %.3f)$ "%mu
                else:
                    if data_type == 'monopole':
                        lin_label = r'$P^\mathrm{EH}_{\ell=0}(k)$'
                        P_label = r"$P_{\ell=0}(k)$"
                    else:
                        lin_label = r'$P^\mathrm{EH}_{\ell=2}(k)$'
                        P_label = r"$P_{\ell=2}(k)$"
                pfy.loglog(ax, k, norm, c='k', label=lin_label)
                pfy.errorbar(ax, k, Pk-Pshot_sub, err, label=P_label, **plot_kwargs)
            
            # normalize by the linear theory
            else:
                
                data_to_plot = (k, (Pk-Pshot_sub)/norm, err/norm)
                pfy.plot_data(ax, data=data_to_plot, labels=label, y_offset=y_offset, plot_kwargs=plot_kwargs)
        
        # just plot the measurement with no normalization
        else:
            
            # set up the labels and plot the measurement
            if data_type == "Pkmu":
                if label is None: label = r"$P(k, \mu = %.3f)$ "%mu
            else:
                if data_type == 'monopole':
                    if label is None: label = r"measured $P_0(k)$"
                else:
                    if label is None: label = r"measured $P_2(k)$"
            pfy.errorbar(ax, k, (Pk - Pshot_sub), err, label=label, **plot_kwargs)
            ax.x_log_scale()
            ax.y_log_scale()

        # now determine the units
        if self.output_units == 'relative':
            k_units = "(h/Mpc)"
            P_units = "(Mpc/h)^3"
        else:
            k_units = "(1/Mpc)"
            P_units = "(Mpc)^3"
        
        # let's set the xlabels and return
        if ax.xlabel.text == "":
            ax.xlabel.update(r"$\mathrm{k \ %s}$" %k_units)
            
        if ax.ylabel.text == "":
            
            if data_type == "Pkmu":
                norm_label = r"P^\mathrm{EH}(k, \mu)"
                if mu_avg: 
                    P_label = r"\langle P(k, \mu) \rangle_\mu"
                else:
                    P_label = r"P(k, \mu)"
            else:
                norm_label = r"P^\mathrm{EH}_{\ell=0}(k)"
                if data_type == "monopole":
                    P_label = r"P_{\ell=0}(k)"
                else:
                    P_label = r"P_{\ell=2}(k)"
            
            if norm_linear:
                if subtract_shot_noise:
                    ax.ylabel.update(r"$\mathrm{(%s - \bar{n}^{-1}) \ / \ %s}$" %(P_label, norm_label))
                else:
                    ax.ylabel.update(r"$\mathrm{%s \ / \ %s}$" %(P_label, norm_label))
            
            else:    
                ax.ylabel.update(r"$\mathrm{%s \ %s}$" %(P_label, P_units))

        return ax.get_figure(), ax
    
    #end plot   
    #---------------------------------------------------------------------------

#endclass PkmuMeasurement
#-------------------------------------------------------------------------------
        
        
