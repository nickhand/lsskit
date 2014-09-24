"""
 pkmu_measurement.py
 class to hold a P(k, mu) measurement
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 07/18/2014
"""
import numpy as np
from pandas import HDFStore, DataFrame, Index, MultiIndex
import itertools
import pickle 

from . import tsal, tools
import plotify as pfy
from cosmology.growth import Power, growth_rate
from cosmology.parameters import Cosmology
from cosmology.utils.units import h_conversion_factor

#-------------------------------------------------------------------------------
def average_power(files, output_units=None, mu_avg=False, mu=None, data_type="Pkmu"):
    """
    Compute the average power from a list of PkmuMeasurement file names
    """
    if output_units is not None:
        if output_units not in ['absolute', 'relative']:
            raise ValueError("`output_units` keyword must be one of ['absolute', 'relative']")
    
    if data_type not in ['Pkmu', 'monopole', 'quadrupole']:
         raise ValueError("Must specify a `type` positional argument and it must be one of ['Pkmu', 'monopole', 'quadrupole']")
    
    frames = []
    for f in files:
        data = load(f)
        if output_units is not None: data.output_units = output_units
    
        if data_type == "Pkmu":
            if mu_avg:
                frames.append(data.mu_averaged())
            else:
                frames.append(data.Pk(mu))
        else:
            frames.append(getattr(data, data_type))

    return tools.weighted_average(frames)
#end average_power

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
    try:
        return pickle.load(open(filename, 'r'))
    except:
        store = HDFStore(filename, 'r')
        pkmu = store.get_storer('data').attrs.Pkmu
        pkmu.data = store['data']
        store.close()
        return pkmu
        
#end load

#-------------------------------------------------------------------------------
class PkmuMeasurement(tsal.TSAL):
    """
    Class to hold a P(k, mu) measurement, which is essentially a wrapper around
    a ``pandas.DataFrame`` object which holds the P(k, mu) data
    """
    def __init__(self, cpm_tsal, units, poles_tsal=None, **kwargs):
        """
        Initialize with the ``pandas.DataFrame`` holding the P(k, mu) data.
        
        Notes
        -----
        `self.data` is a `DataFrame` holding the P(k, mu) measurement,
        which will have a `MultiIndex` with levels `[`mu`, `k`]`, which defines
        each measurement. The columns for the `DataFrame` are:
            `power` : the comoving power measurement with units specified
                      by `units`
            `error` : the error on the power measurement
            `baseline` : the baseline power measurement that the power 
                         is measured with respect to
            `noise` : the component of the baseline that is "noise-like"
        
        Parameters
        ----------
        cpm_tsal : str
            The name of the file holding the P(k,mu) measurement in the 
            form of a TSAL

        
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
                        
            space : str, {`real`, `redshift`, `None`}
                Whether the measurement was made in real or redshift space. 
                Default is `None`
        """
        # dictionary storing the cosmology
        self.cosmo = kwargs.pop('cosmo', None)
        if self.cosmo is not None:
            if not isinstance(self.cosmo, (dict, basestring, Cosmology)): 
                raise TypeError("The cosmology object must be a dict/string/Cosmology")
            if isinstance(self.cosmo, basestring):
                self.cosmo = Cosmology(self.cosmo)

        # the measurement units
        if units not in ['absolute', 'relative']:
            raise ValueError("`Units` must be one of [`absolute`, `relative`]")
        self.measurement_units = units
        
        # keep track of the units for output
        self._output_units = self.measurement_units

        # the space of the measurement
        self.space = kwargs.pop('space', None)

        # any other keywords we want to save
        for k, v in kwargs.iteritems():
            setattr(self, k, v)        
            
        # the data frame storing the P(k, mu) data
        self._cpm_from_tsal(cpm_tsal)

        # add multipoles too
        if poles_tsal is not None:
            self._poles_from_tsal(poles_tsal)
            
        # add the growth factor for convenience
        self.f = growth_rate(self.redshift, params=self.cosmo)
    #end __init__

    #---------------------------------------------------------------------------
    def __getitem__(self, key):
        """
        Access individual band-powers through dict-like interface:

            Pkmu_frame = self[k, mu], 

        where k and mu are either integers between 0 - len(self.ks)/len(self.mus) 
        or floats specifying the actual value

        Returns
        -------
        frame : pandas.DataFrame
            A DataFrame with a single row holding the data for the band, 
            which columns specified by `self.columns`
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Must specify both a `k` and `mu` value as the key")

        k, mu = key
        if isinstance(k, int): 
            if not 0 <= k < len(self.ks): 
                raise KeyError("Integer that identifies k-band must be between [0, %d)" %len(self.ks))
            k = self.ks[k]

        if isinstance(mu, int): 
            if not 0 <= mu < len(self.mus): 
                raise KeyError("Integer that identifies mu-band must be between [0, %d)" %len(self.mus))
            mu = self.mus[mu]

        if not (mu, k) in self.data.index:
            raise KeyError("Sorry, band (k = %s, mu = %s) does not exist" %(k, mu))

        return self.data.loc[mu, k]

    #end __getitem__

    #---------------------------------------------------------------------------
    # Internal methods
    #---------------------------------------------------------------------------
    def _cpm_from_tsal(self, fname):
        """
        Internal method to read the CPM TSAL file, optionally reading any 
        extra components
        """
        ff = open(fname)
        self.readTSAL(ff) # inherited function
        self._read_extra_cpm(ff)
        ff.close()

    #end _cpm_from_tsal

    #---------------------------------------------------------------------------
    def _read_extra_cpm(self, ff):
        """
        Internal method to read extra components of the TSAL
        """
        self.redshift = float(ff.readline()) # redshift

        # check syntax
        line = ff.readline()
        if line != "add_to_baseline\n" :
            print line
            raise ValueError("no add_to_baseline - Not sure what to do in _read_extra")

        # read the baseline
        baseline = {}
        ks       = set()
        mus      = set()
        N        = int(ff.readline())
        for i in range(N):
            k, mu, val = map(float, ff.readline().split())
            ks.add(k)
            mus.add(mu)
            baseline[(k,mu)] = val

        # check syntax
        line = ff.readline()        
        if line != "approx_noise\n" :
            print line
            raise ValueError("no approx_noise - Not sure what to do in _read_extra")

        # check syntax
        if N != int(ff.readline()) :
            raise ValueError("number mismatch in _read_extra")

        # read the noise 
        noise = {}
        for i in range(N):
            k, mu, val = map(float, ff.readline().split())
            if (k, mu) not in baseline:
                print i, k, mu
                raise ValueError("k or mu mismatch in _read_extra")
            noise[(k,mu)] = val

        # all combinations of (mu, k)
        muks = list(itertools.product(sorted(mus), sorted(ks)))

        # the column values for each (mu, k)
        columns = []
        base_name = '_'.join(self.pars.keys()[0].split('_')[:2])
        for (mu, k) in muks:
            this_base  = baseline[(k,mu)]
            this_noise = noise[(k,mu)]
            name  = "%s_%s_%s" %(base_name, str(k), str(mu))
            this_val   = this_base + self.pars[name].val
            this_err   = self.pars[name].err           
            columns.append( (this_val,this_err,this_noise,this_base) )

        # now make the DataFrame
        index = MultiIndex.from_tuples(muks, names=['mu', 'k'])
        frame = DataFrame(columns, index=index, columns=['power', 'error', 'noise', 'baseline'])

        # save as `data` attribute
        self.data = frame

    #end _read_extra_cpm

    #---------------------------------------------------------------------------
    def _poles_from_tsal(self, tsal_file):
        """
        Internal function to store multipoles from TSAL file
        """
        # read the multipoles TSAL
        tsal_fit = tsal.TSAL(tsal_file)

        mono, quad, hexadec = {}, {}, {}
        for key, val in tsal_fit.pars.iteritems():

            k = float(key.split('_')[-1])
            if 'mono' in key:
                mono[k] = (val.val, val.err)
            elif 'quad' in key:
                quad[k] = (val.val, val.err)
            elif 'hexadec' in key:
                hexadec[k] = (val.val, val.err)
            else:
                raise ValueError("Multipoles TSAL name %s not recognized" %key)

        # set up the index for multipole data frames 
        ks = np.array(sorted(mono.keys()))
        if np.any(self.ks != ks):
            raise ValueError("Wavenumber mismatch between P(k, mu) and multipole measurements")
        index = Index(self.ks, name='k')    

        # now make the data frames
        names = ['_monopole', '_quadrupole', '_hexadecapole']
        for name, pole in zip(names, [mono, quad, hexadec]):
            if len(pole) != len(self.ks):
                continue
            vals, errs = map(np.array, zip(*[pole[k] for k in ks]))
            frame = DataFrame(data=np.vstack((vals, errs)).T, index=index, columns=['power', 'error'])
            setattr(self, name, frame)

    #end _poles_from_tsal
    
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
        Return a `DataFrame` with the `power`, `error` columns storing the
        monopole values and errors, respectively, in units specified by 
        `self.output_units`
        
        Returns
        -------
        df : pandas.DataFrame
            The monopole data in DataFrame format
        """
        # conversion factor for output units, as specified in `self.output_units`
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('power', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])
        try:
            return output_units_factor*self._monopole
        except:
            raise AttributeError("No monopole measurement provided yet.")
    #end monopole
    
    #---------------------------------------------------------------------------
    @property
    def quadrupole(self):
        """
        Return a `DataFrame` with the `power`, `error` columns storing the
        quadrupole values and errors, respectively, in units specified by 
        `self.output_units`
        
        Returns
        -------
        df : pandas.DataFrame
            The quadrupole data in DataFrame format
        """
        # conversion factor for output units, as specified in `self.output_units`
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('power', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])
        try:
            return output_units_factor*self._quadrupole
        except:
            raise AttributeError("No quadrupole measurement provided yet.")
    #end quadrupole

    #---------------------------------------------------------------------------
    @property
    def hexadecapole(self):
        """
        Return a `DataFrame` with the `power`, `error` columns storing the
        hexadecapole values and errors, respectively, in units specified by 
        `self.output_units`
        
        Returns
        -------
        df : pandas.DataFrame
            The quadrupole data in DataFrame format
        """
        # conversion factor for output units, as specified in `self.output_units`
        output_units_factor = 1.
        if self.measurement_units != self.output_units:
            output_units_factor = h_conversion_factor('power', self.measurement_units, 
                                                        self.output_units, self.cosmo['h'])
        try:
            return output_units_factor*self._hexadecapole
        except:
            raise AttributeError("No hexadecapole measurement provided yet.")
    #end quadrupole
    
    #---------------------------------------------------------------------------
    # the functions        
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
                                                        
        # get the correct value of mu, if an integer was specified
        if isinstance(mu, int): 
            if not 0 <= mu < len(self.mus): 
                raise KeyError("Integer that identifies mu-band must be between [0, %d)" %len(self.mus))
            mu = self.mus[mu]
            
        # make sure the power is defined at this value of mu
        if mu not in self.mus:
            raise ValueError("Power measurement not defined at mu = %s" %mu)
        
        return self.data.xs(mu, level=self._mu_level)*output_units_factor
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
                                                        
        # get the correct value of k, if an integer was specified
        if isinstance(k, int): 
            if not 0 <= k < len(self.ks): 
                raise KeyError("Integer that identifies k-band must be between [0, %d)" %len(self.ks))
            k = self.ks[k]
            
        # make sure the power is defined at this value of k
        if k not in self.ks:
            raise ValueError("Power measurement not defined at k = %s" %k)
        
        return self.data.xs(k, level=self._k_level)
    #end Pmu
    
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
            err_average = lambda row: np.sum(row['error']**(-2.))
        
            avg_data['power'] = grouped.apply(power_average)
            avg_data['error'] = (grouped.apply(err_average))**(-0.5)
        else:
            avg_data['error'] /= np.sqrt(len(self.mus))
            
        return avg_data*output_units_factor
    #end mu_averaged
    
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
            ks = self.ks * h_conversion_factor('wavenumber', self.output_units, 'relative', self.cosmo['h'])
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
        the units specified by `self.output_units`.  In addition to the 
        keywords accepted below, any plotting keywords accepted by 
        `matplotlib` can be passed.

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
            class. Default is `{'transfer_fit' : "EH"}`.
            
        subtract_constant_noise : bool, optional
            If `True`, subtract the 1/nbar shot noise stored in `Pshot` before 
            plotting. Default is `False`. This assumes you can compute the 
            shot noise from the `self.Pshot`.

        subtract_noise : bool, optional
            If `True`, subtract the noise component stored as the `noise`
            column in `self.data`. Default is `False`. 

        normalize_by_baseline : bool, optional
            Normalize by the baseline, plotting data/baseline. This is only
            possible for the `Pkmu` `data_type`. Default is `False`
            
        label : str, optional
            The label to attach to these plot lines. Default is `None`
        
        offset : float, optional
            Offset the plot in the y-direction by this amount. Default is `0`
        
        weighted_mean : bool, optional
            When averaging the power spectrum over `mu`, whether or not
            to use a weighted average. Default is `False`
            
        add_axis_labels : bool, optional
            Whether to add axis labels that make sense. Default is `True`
        
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
        sub_constant_noise = kwargs.pop('subtract_constant_noise', False)
        sub_noise          = kwargs.pop('subtract_noise', False)
        bias               = kwargs.pop('bias', 1.)
        norm_linear        = kwargs.pop('norm_linear', False)
        plot_linear        = kwargs.pop('plot_linear', False)
        power_kwargs       = kwargs.pop('power_kwargs', {'transfer_fit' : "EH"})
        label              = kwargs.pop('label', None)
        y_offset           = kwargs.pop('offset', 0.)
        weighted_mean      = kwargs.pop('weighted_mean', False)
        add_axis_labels    = kwargs.pop('add_axis_labels', True)
        normalize          = kwargs.pop('normalize_by_baseline', False)

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
        data_choices = ['Pkmu', 'monopole', 'quadrupole', 'hexadecapole']
        if data_type not in data_choices:
            raise ValueError("Positional argument must be one of %s" %data_choices)
            
        # now get the data
        k = self.ks 
        norm = 1.
        pole_numbers = {'monopole':'0', 'quadrupole':'2', 'hexadecapole':'4'}
        if data_type == 'Pkmu':
            # mu-averaged or not
            mu = kwargs.pop('mu', None)
            mu_avg = mu is None
            if mu_avg:
                mu   = np.mean(self.mus)
                data = self.mu_averaged(weighted=weighted_mean)
            else:
                # get the mu value, correctly 
                if isinstance(mu, int): 
                    if not 0 <= mu < len(self.mus): 
                        raise KeyError("Integer that identifies mu-band must be between [0, %d)" %len(self.mus))
                    mu = self.mus[mu]
                data = self.Pk(mu)
        else:
            mu = mu_avg = None
            data = getattr(self, data_type)
 
        Pk   = data.power
        err  = data.error 
        if normalize: norm = data.baseline
        
        # the shot noise
        noise = np.zeros(len(Pk))
        if sub_noise: 
            noise = data.noise
        elif sub_constant_noise: 
            noise = self.Pshot
          
        # plot the linear theory result
        if plot_linear or norm_linear:

            # the normalization
            if norm_linear and data_type in ['quadrupole', 'hexadecapole']:
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
                    lin_label = r'$P^\mathrm{EH}_{\ell=%s}(k)$' %(pole_numbers[data_type])
                    P_label = r"$P_{\ell=%s}(k)$" %(pole_numbers[data_type])
  
                pfy.loglog(ax, k, norm, c='k', label=lin_label)
                pfy.errorbar(ax, k, Pk-noise, err, label=P_label, **kwargs)
            
            # normalize by the linear theory
            else:
                data_to_plot = (k, (Pk-noise)/norm, err/norm)
                pfy.plot_data(ax, data=data_to_plot, labels=label, y_offset=y_offset, plot_kwargs=kwargs)
        
        # just plot the measurement with no normalization
        else:
            
            if label is None:            
                if data_type == "Pkmu":
                    label = r"$P(k, \mu = %.3f)$ "%mu
                else:
                    label = r"$P_{\ell=%s}(k)$" %(pole_numbers[data_type])
            pfy.errorbar(ax, k, (Pk - noise)/norm, err/norm, label=label, **kwargs)
            if not normalize:
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
                P_label = r"P(k, \mu)"
                if not normalize:
                    norm_label = r"P^\mathrm{EH}(k, \mu)"
                else:
                    norm_label = r"P_\mathrm{base}(k, \mu)"

                if mu_avg: 
                    P_label = r"\langle %s \rangle_\mu" %P_label
                    norm_label = r"\langle %s \rangle_\mu" %norm_label
            else:
                norm_label = r"P^\mathrm{EH}_{\ell=%s}(k)" %(pole_numbers[data_type])
                P_label = r"P_{\ell=%s}(k)" %(pole_numbers[data_type])
            
            if norm_linear or normalize:
                if sub_noise or sub_constant_noise:
                    ax.ylabel.update(r"$\mathrm{(%s - \bar{n}^{-1}) \ / \ %s}$" %(P_label, norm_label))
                else:
                    ax.ylabel.update(r"$\mathrm{%s \ / \ %s}$" %(P_label, norm_label))
            
            else: 
                if sub_noise or sub_constant_noise:
                    ax.ylabel.update(r"$\mathrm{%s - \bar{n}^{-1} \ %s}$" %(P_label, P_units))
                else:       
                    ax.ylabel.update(r"$\mathrm{%s \ %s}$" %(P_label, P_units))

        return ax.get_figure(), ax
    
    #end plot   
    #---------------------------------------------------------------------------
    def save(self, filename):
        """
        Save the `PkmuMeasurement` instance as a pickle to the filename specified
        
        Parameters
        ----------
        filename : str 
            the filename to output to
        """        
        pickle.dump(self, open(filename, 'w'))
    #end save
    #---------------------------------------------------------------------------
    
#endclass PkmuMeasurement
#-------------------------------------------------------------------------------
        
        
