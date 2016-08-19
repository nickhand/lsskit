"""
 power_measurement.py 
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 09/26/2014
"""
import numpy as np
from pandas import HDFStore, DataFrame, Index, MultiIndex, Series
import itertools
import pickle 
import copy

from . import tsal, tools
import plotify as pfy
from cosmology.utils.units import h_conversion_factor
from pyRSD import pygcl

#-------------------------------------------------------------------------------
def read_power_measurements(pattern, subtract_shot_noise=True, output_units='relative'):
    """
    Read the power files specified by `pattern`
    
    Parameters
    ----------
    pattern : str 
        The pattern specifying files holding a `PowerMeasurements` object
    subtract_shot_noise : bool, optional
        Whether to subtract the shot noise from the power measurement. Default
        is `True`
    output_units : {'relative', 'absolute'}, optional
        The output units to use. Default is `relative`
        
    Returns
    -------
    data : power_measurement.PkmuMeasurement
        a `PkmuMeasurement` object holding the weighted mean of the input files
    """
    files = tools.glob(pattern)
    if len(files) == 0:
        raise ValueError("No files match the pattern specified")
    
    # load the data 
    data = []
    for f in files:
        d = load(f)
        if not isinstance(d, PowerMeasurements):
            raise TypeError("Files must hold a `PowerMeasurements` object")
        d  = d['pkmu']
        
        # now format
        d.subtract_shot_noise = subtract_shot_noise
        d.output_power_units = output_units
        d.output_k_units = output_units
        data.append(d)
            
    return tools.weighted_mean(data)

#-------------------------------------------------------------------------------
def read_pole_measurements(pattern, subtract_shot_noise=True, output_units='relative'):
    """
    Read the power files specified by `pattern`
    
    Parameters
    ----------
    pattern : str 
        The pattern specifying files holding a `PowerMeasurements` object
    subtract_shot_noise : bool, optional
        Whether to subtract the shot noise from the monopole measurement. Default
        is `True`
    output_units : {'relative', 'absolute'}, optional
        The output units to use. Default is `relative`
        
    Returns
    -------
    mono : power_measurement.MonopoleMeasurement
        a `MonopoleMeasurement` object holding the weighted mean of the input
    quad : power_measurement.QuadrupoleMeasurement
        a `QuadrupoleMeasurement` object holding the weighted mean of the input
    """
    files = tools.glob(pattern)
    if len(files) == 0:
        raise ValueError("No files match the pattern specified")

    # load the data 
    monos = []
    quads = []
    for f in files:        
        d = load(f)
        if not isinstance(d, PowerMeasurements):
            raise TypeError("Files must hold a `PowerMeasurements` object")
        
        mono = d['monopole']
        mono.subtract_shot_noise = subtract_shot_noise
        mono.output_power_units = output_units
        mono.output_k_units = output_units
        monos.append(mono)

        quad = d['quadrupole']
        quad.output_power_units = output_units
        quad.output_k_units = output_units
        quads.append(quad)


    return tools.weighted_mean(monos), tools.weighted_mean(quads)
    
#-------------------------------------------------------------------------------
def write_multipoles(filename, mono=None, quad=None, hexadec=None, remove_missing=False):
    """
    Save the power multipole measurements to a data file in ASCII format
    
    Parameters
    ----------
    filename : str
        The name of the file to write to
    mono : MonopoleMeasurement 
        The monopole measurement object
    quad : QuadrupoleMeasurement
        The quadrupole measurement object
    hex : HexadecapoleMeasurement
        The hexadecapole measurement object
    remove_missing : bool, optional
        whether to remove any missing data points
    """ 
    poles = {}
    if mono is not None:
        poles['monopole'] = mono
    if quad is not None:
        poles['quadrupole'] = quad
    if hexadec is not None:
        poles['hexadecapole'] = hexadec
    
    if len(poles) == 0:
        raise ValueError("Must supply at least one multipole measurement")
    
    # check the output units
    k_units = [x.output_k_units for x in poles.values()]
    if not all(k_units[0] == item for item in k_units):
        raise ValueError("multipole measurements must have same output k units")
        
    power_units = [x.output_power_units for x in poles.values()]
    if not all(power_units[0] == item for item in power_units):
        raise ValueError("multipole measurements must have same output power units")
   
    # remove missing?
    if remove_missing:
        for name in poles:
            poles[name]._data = poles[name]._data.dropna()
    
    name0 = poles.keys()[0]
    pole0 = poles[name0]
    k_units = "h/Mpc" if pole0.output_k_units == 'relative' else "1/Mpc"
    power_units = "(Mpc/h)^3" if pole0.output_power_units == 'relative' else "(Mpc)^3"
    
    shot_noise_hdr = ""
    if 'monopole' in poles and poles['monopole'].subtract_shot_noise:
        Pshot = poles['monopole'].shot_noise
        shot_noise_hdr = "shot noise subtracted from monopole: P_shot = %.5e %s\n" %(Pshot, power_units)
    
    names = ", ".join(poles.keys()) if len(poles) > 1 else name0
    header = "%s in %s space at redshift z = %s\n" %(names, pole0.space, pole0.redshift) + \
             "for k_cen = %.5f to %.5f %s,\n" %(np.amin(pole0.ks), np.amax(pole0.ks), k_units) + \
             "number of wavenumbers equal to %d\n" %(len(pole0.ks)) + \
             shot_noise_hdr
             
    cols = "%s1:k_cen (%s)%s" %(" "*5, k_units, " "*5)
    for i, name in enumerate(poles):
        cols += "%d:%s %s%s%d:error %s%s" %(2*i+1, name, power_units, " "*5, 
                                            2*(i+1), power_units, " "*5)
    header += cols
    
    data = (pole0.ks,)
    for name in poles:
        pole = poles[name]
        data += (pole.data.power, pole.data.variance**0.5)
    toret = np.vstack(data).T
    np.savetxt(filename, toret, header=header, fmt="%20.5e")

#-------------------------------------------------------------------------------
def load(filename):
    """
    Load a ``PkmuMeasurements`` object from a pickle file
    
    Parameters
    ----------
    filename : str
        The name of the pickled file
    
    Returns
    -------
    pkmu : PkmuMeasurements object
        The PkmuMeasurements class holding the power measurements
    """
    try:
        return pickle.load(open(filename, 'r'))
    except:
        store = HDFStore(filename, 'r')
        power = store.get_storer('data').attrs.power
        power._data = store['data']
        store.close()
        return power
        
#end load

#-------------------------------------------------------------------------------
class PowerMeasurement(tsal.TSAL):
    """
    Base class to hold a power measurement, which is either P(k, mu) or a power
    multipole. The class is essentially a wrapper around a ``pandas.DataFrame`` 
    object which holds measurement in the `data` attribute
    """
    def __init__(self, tsal_file, units, **kwargs):
        """
        Initialize the class with the TSAL file and keywords
        """
        # the cosmology
        self.cosmo = kwargs.pop('cosmo', None)
        if self.cosmo is not None:
            if not isinstance(self.cosmo, (basestring, pygcl.Cosmology)): 
                raise TypeError("`Cosmo` must be one of [dict, str, pygcl.Cosmology]")
            if isinstance(self.cosmo, basestring): 
                self.cosmo = pygcl.Cosmology(self.cosmo)

        # the measurement units
        if units not in ['absolute', 'relative']:
            raise ValueError("`Units` must be one of [`absolute`, `relative`]")
        self.measurement_units = {'k' : units, 'power' : units}
        
        # keep track of the units for output k/power
        self._output_k_units = units
        self._output_power_units = units

        # the space of the measurement
        self.space = kwargs.pop('space', None)

        # any other keywords we want to save
        for k, v in kwargs.iteritems(): 
            try:
                setattr(self, k, v)        
            except:
                print(k, v)
        
        # the load the data as a tsal
        if tsal_file is not None: self._data_from_tsal(tsal_file)

        # add shot noise column to self.data
        self._add_shot_noise_column()
        
        # add the growth factor for convenience
        if self.cosmo is not None and hasattr(self, 'redshift'):
            try:
                self.f = self.cosmo.f_z(self.redshift)
            except:
                self.f = None
                    
        # normalize attribute determines whether or not to return 
        # the normalized power 
        self._normalize = False
        
        # subtract shot noise from the return power
        self._subtract_shot_noise = False
        
        # keep track of read-only attributes (normalize, subtract_shot_noise, output_units)
        self._read_only = set()
        
        # delete unneccesary (and potentially large TSAL base parameters)
        for att in ['pars', 'sd', 'fd', 'mat']:
            if hasattr(self, att): delattr(self, att)
    #end __init__
    
    #----------------------------------------------------------------------------
    # Internal methods
    #---------------------------------------------------------------------------
    def __abs__(self):
        """
        Absolute value
        """
        toret = self.copy()
        toret._data = abs(toret._data)
        return toret
        
    #---------------------------------------------------------------------------
    def __add__(self, other):
        """
        Operator overloading for addition
        """
        toret = self.copy()
                    
        # if adding two PowerMeasurements, keep track of variance correctly
        if isinstance(other, self.__class__):
            
            # check the units
            this_units = None if self.normalize else self.output_power_units
            other_units = None if other.normalize else other.output_power_units
            if this_units != other_units:
                raise TypeError("Cannot add `PowerMeasurement` objects with different output power units")
            
            this_normalized = True if this_units is None else self.normalize
            other_normalized = True if other_units is None else other.normalize
            if this_normalized != other_normalized:
                raise ValueError("Bad, bad idea to add two measurements while only one is normalized")
            
            # do the actual addition
            toret._data = self.data + other.data
            toret._data['variance'] = self._add_variances(self.data, other.data)
        
            # power measurement units returned are output units of added
            toret.measurement_units['power'] = self.output_power_units
                
            # make sure we can't mess up the output
            if self.normalize: 
                toret.update(_normalize=False)                
                toret._read_only.add('normalize')

                # if we normalized, units are None and we can't change them
                toret._read_only.add('output_power_units')
                toret.measurement_units['power'] = toret._output_power_units = None
        else:
            try:
                # add the data frames, noting that adding a "well-defined"
                # value to the measurement doesn't change the variance
                toret._data = self._data.add(other, axis='index')
                toret._data['variance'] = self._data['variance']
            except:
                raise ValueError("Failure to add object of type %s" %type(other))
                           
        return toret
        
    def __radd__(self, other):
        return self + other
        
    #---------------------------------------------------------------------------
    def __sub__(self, other):
        """
        Operator overloading for subtraction
        """
        toret = self.copy()
        
        # if subtracting two PowerMeasurements, keep track of variance correctly
        if isinstance(other, self.__class__):
            
            # check the units
            this_units = None if self.normalize else self.output_power_units
            other_units = None if other.normalize else other.output_power_units
            if this_units != other_units:
                raise TypeError("Cannot subtract `PowerMeasurement` objects with different output power units")
            
            this_normalized = True if this_units is None else self.normalize
            other_normalized = True if other_units is None else other.normalize
            if this_normalized != other_normalized:
                raise ValueError("Bad, bad idea to add two measurements while only one is normalized")
            
            # do the subtraction
            toret._data = self.data - other.data
            toret._data['variance'] = self._add_variances(self.data, other.data)
            
            # power measurement units returned are output units of added
            toret.measurement_units['power'] = self.output_power_units
                                
            # make sure we can't mess up the output
            if self.normalize: 
                toret.update(_normalize=False)                
                toret._read_only.add('normalize')

                # if we normalized, units are None and we can't change them
                toret._read_only.add('output_power_units')
                toret.measurement_units['power'] = toret._output_power_units = None
        else:
            try:
                # subtract the data frames, noting that subtracting a "well-defined"
                # value to the measurement doesn't change the variance
                toret._data = self._data.subtract(other, axis='index')
                toret._data['variance'] = self._data['variance']
            except:
                raise ValueError("Failure to subtract object of type %s" %type(other))
                           
        return toret

    def __rsub__(self, other):
        return (self - other) * -1.
        
    #---------------------------------------------------------------------------
    def __div__(self, other):
        """
        Operator overloading for division
        """
        toret = self.copy()
    
        # if dividing two PowerMeasurements, keep track of variance correctly
        if isinstance(other, self.__class__):
            
            # check the units
            this_units = None if self.normalize else self.output_power_units
            other_units = None if other.normalize else other.output_power_units
            if this_units != other_units:
                raise TypeError("Cannot divide `PowerMeasurement` objects with different output power units")
            
            this_normalized = True if this_units is None else self.normalize
            other_normalized = True if other_units is None else other.normalize
            if this_normalized != other_normalized:
                raise ValueError("Bad, bad idea to add two measurements while only one is normalized")
                                
            # do the actual division
            toret._data = toret.data/other.data
            var_sum = self._add_fractional_variances(self.data, other.data)
            toret._data['variance'] = (self.data.power/other.data.power)**2*var_sum
            
            # make sure we can't mess up the output
            toret.update(_normalize=False, _subtract_shot_noise=False)                
            for x in ['normalize', 'subtract_shot_noise', 'output_power_units']:
                toret._read_only.add(x)
           
            # set the shot noise to zero
            toret._data['shot_noise'] *= 0.            
            # returned power doesn't have units
            toret.measurement_units['power'] = toret._output_power_units = None
                
        else:
            try:
                # divide by the constant, dividing the variance by the square
                # of `other`
                toret._data = self._data.divide(other, axis='index')
                toret._data['variance'] = self._data['variance'].divide(other, axis='index')
            except:
                raise ValueError("Failure to divide object of type %s" %type(other))
                    
        return toret
        
    #---------------------------------------------------------------------------
    def __mul__(self, other):
        """
        Operator overload for division
        """
        toret = self.copy()
    
        # if multiplying two PowerMeasurements, keep track of variance correctly
        if isinstance(other, self.__class__):
            raise NotImplementedError("Not a good idea to multiply two PowerMeasurements together")
            
        else:
            try:
                # multiply by the constant, multiplying the variance by the square
                # of `other`
                toret._data = self._data.multiply(other, axis='index')
                toret._data['variance'] = self._data['variance'].multiply(other, axis='index')
            except:
                raise ValueError("Failure to multiply object of type %s" %type(other))
                    
        return toret
        
    def __rmul__(self, other):
        return self*other
        
    #---------------------------------------------------------------------------
    def _add_shot_noise_column(self):
        """
        Add the shot noise to `self.data` as column named `shot_noise`
        """
        if hasattr(self._data, 'shot_noise'):
            return
            
        Pshot = 0.
        if hasattr(self, 'volume') and hasattr(self, 'sample_size'):
            Pshot = (self.volume / self.sample_size)
        elif hasattr(self, 'Pshot'):
            Pshot = self.Pshot
        elif hasattr(self, 'shot_noise'):
            Pshot = self.shot_noise
            
        self._data['shot_noise'] = (self._data['power']*0. + Pshot)
    #end _add_shot_noise_column
    
    #---------------------------------------------------------------------------
    def _add_fractional_variances(self, this, that):
        """
        Internal method to compute variance by adding the fractional
        variances
        """
        fvar1 = this.variance/this.power**2
        fvar2 = that.variance/that.power**2
        return fvar1 + fvar2
        
    #end _add_fractional_variances
    
    #---------------------------------------------------------------------------
    def _add_variances(self, this, that):
        """
        Internal method to compute variance by adding the fractional
        variances
        """
        return this.variance + that.variance
        
    #end _add_variances
    
    #---------------------------------------------------------------------------
    def _data_from_tsal(self, fname):
        """
        Internal method to read a `ComovingPowerMeasurement` TSAL file, 
        which involves reading the extra components
        """
        ff = open(fname)
        self.readTSAL(ff)
        lines = ff.readlines()
        self.redshift = float(lines[0]) # redshift

        #-----------------------------------------------------------------------
        # Read the extra quantities
        #-----------------------------------------------------------------------
        ks, mus = set(), set()
        extras = {'baseline' : {}, 'noise' : {}, 'modes' : {}}
        names = {'baseline' : 'add_to_baseline\n', 'noise' : 'approx_noise\n',
                 'modes' : 'mode_counts\n'}
        for extra in extras:
            if not names[extra] in lines:
                if extra in ['baseline', 'noise']:
                    raise ValueError("no %s to read -- not sure what to do in `_data_from_tsal`")
                continue
            
            index = lines.index(names[extra])
            for i in range(int(lines[index+1])):
                k, mu, val = map(float, lines[index+2+i].split())
                if extra == 'baseline':
                    ks.add(k)
                    mus.add(mu)
                extras[extra][(k,mu)] = val

        # read the poisson shot noise
        Pshot = 0
        if 'poisson_shot_noise\n' in lines:
            index = lines.index('poisson_shot_noise\n')
            Pshot = float(lines[index+1])

        # all combinations of (mu, k)
        muks = list(itertools.product(sorted(mus), sorted(ks)))

        # the column values for each (mu, k)
        columns = []
        base_name = '_'.join(self.pars.keys()[0].split('_')[:2])
        for (mu, k) in muks:
            this_base  = extras['baseline'][(k,mu)]
            this_noise = extras['noise'][(k,mu)]
            this_Nmodes = extras['modes'][(k,mu)] if len(extras['modes']) > 0 else 0.
            
            if (k % 1) == 0: k = int(k)
            if (mu % 1) == 0: mu = int(mu)
            name  = "%s_%s_%s" %(base_name, str(k), str(mu))
            this_val   = this_base + self.pars[name].val
            this_err   = self.pars[name].err           
            columns.append( (this_val, this_err**2, this_noise, this_base, this_Nmodes) )

        # now make the DataFrame
        index = MultiIndex.from_tuples(muks, names=['mu', 'k'])
        self._data = DataFrame(columns, index=index, columns=['power', 'variance', 'noise', 'baseline', 'modes'])

        # now remove any unmeasured modes (will have baseline==power)
        to_replace = self._data.power == self._data.baseline
        self._data.loc[to_replace, ['power', 'variance']] = np.NaN
        ff.close()
        
        # add the shot noise column if we read it
        if Pshot > 0:
            self._data['shot_noise'] = (self._data['power']*0. + Pshot)

    #end _data_from_tsal

    #---------------------------------------------------------------------------
    def _h_conversion_factor(self, data_type, from_units, to_units):
        """
        Internal method to compute units conversions
        """
        factor = 1.
        if from_units is None or to_units is None:
            return factor
        
        if to_units != from_units:
            factor = h_conversion_factor(data_type, from_units, to_units, self.cosmo['h'])
            
        return factor
    #end _h_conversion_factor
    
    #---------------------------------------------------------------------------
    def _convert_k_integer(self, k_int):
        """
        Convert the integer value specifying a given `k` to the actual value
        """
        msg = "[0, %d) or [-%d, -1]" %(len(self.ks), len(self.ks))
        if k_int < 0: k_int += len(self.ks)
        if not 0 <= k_int < len(self.ks):
            raise KeyError("Integer that identifies k-band must be between %s" %msg)
        return self.ks[k_int]
        
    #---------------------------------------------------------------------------
    def copy(self):
        """
        Return a deep copy of `self`
        """
        return copy.deepcopy(self)
    #end copy
    
    #---------------------------------------------------------------------------
    def update(self, **kwargs):
        """
        Update the attributes 
        """
        for k, v in kwargs.iteritems(): setattr(self, k, v)
    
    #---------------------------------------------------------------------------
    # Properties
    #---------------------------------------------------------------------------
    @property
    def shot_noise(self):
        """
        The shot noise in units specified by `self.output_power_units`
        """
        factor = self._h_conversion_factor('power', self.measurement_units['power'], self.output_power_units)
        return self._data.shot_noise.dropna().iloc[0]*factor
        
    #---------------------------------------------------------------------------
    @property
    def data(self):
        """
        The `pandas.DataFrame` holding the power measurement, in units
        specified by `self.output_power_units`
        """
        factor = self._h_conversion_factor('power', self.measurement_units['power'], self.output_power_units)
        toret = self._data*factor
        toret['variance'] *= factor
        return toret
    
    #---------------------------------------------------------------------------
    @property
    def output_power_units(self):
        """
        The type of units to use for output power quanities. Either `relative` or 
        `absolute`. Initialized to `self.measurement_units['power']`
        """
        return self._output_power_units
        
    @output_power_units.setter
    def output_power_units(self, val):
        
        if val not in ['absolute', 'relative', None]:
            raise AttributeError("`output_power_units` must be one of [`absolute`, `relative`]")
        
        if val != self.measurement_units['power'] and self.cosmo is None:
            raise AttributeError("Cannot convert units of output power quantities without a cosmology defined")
            
        if 'output_power_units' in self._read_only:
            raise AttributeError("Cannot set `output_power_units`; must be `%s`" %self._output_power_units)
            
        self._output_power_units = val
    
    #---------------------------------------------------------------------------
    @property
    def output_k_units(self):
        """
        The type of units to use for output wavenumber. Either `relative` or 
        `absolute`. Initialized to `self.measurement_units['k']`
        """
        return self._output_k_units
        
    @output_k_units.setter
    def output_k_units(self, val):
        
        if val not in ['absolute', 'relative']:
            raise AttributeError("`output_k_units` must be one of [`absolute`, `relative`]")
        
        if val != self.measurement_units['k'] and self.cosmo is None:
            raise AttributeError("Cannot convert units of output waveumbers without a cosmology defined")
            
        self._output_k_units = val
    
    #---------------------------------------------------------------------------
    @property
    def normalize(self):
        """
        When `data` attribute is accessed, return the `DataFrame` normalized
        by the return of `self.normalization(self.bias)`
        """
        return self._normalize
        
    @normalize.setter
    def normalize(self, val):
        
        if 'normalize' in self._read_only:
            raise AttributeError("Cannot set `normalize`; must be `%s`" %self._normalize)
        self._normalize = val
        
    #---------------------------------------------------------------------------
    @property
    def subtract_shot_noise(self):
        """
        When `data` attribute is accessed, first subtract the column `shot_noise`
        from the `self.data`
        """
        return self._subtract_shot_noise
        
    @subtract_shot_noise.setter
    def subtract_shot_noise(self, val):
        
        if 'subtract_shot_noise' in self._read_only:
            raise AttributeError("Cannot set `subtract_shot_noise`; must be `%s`" %self._subtract_shot_noise)
            
        self._subtract_shot_noise = val
    
    #---------------------------------------------------------------------------
    @property
    def columns(self):
        """
        Return the columns of the ``pandas.DataFrame`` stored in `self.data`
        
        Returns
        -------
        columns : list
             A list of string column names in the `DataFrame` 
        """
        return list(self._data.columns)
    #end columns
        
    #---------------------------------------------------------------------------
    @property
    def ks(self):
        """
        The k values where the measurement is defined, in units specified
        by `self.output_k_units`
        
        Returns
        -------
        ks : np.ndarray
            An array holding the k values at which the power is measured
        """
        factor = self._h_conversion_factor('wavenumber', self.measurement_units['k'], self.output_k_units)
        if isinstance(self._data.index, MultiIndex):
            return np.asarray(self._data.index.levels[self._k_level], dtype=float)*factor
        else:
            return np.asarray(self._data.index, dtype=float)*factor
    #end ks
    
    #---------------------------------------------------------------------------
    def rebin_k(self, N, weighted=True):
        """
        Return a copy of `self` with the `data` DataFrame rebinnned, with
        a total of `N` `k` bins
        
        Parameters
        ----------
        N : int
            The number of k bins to rebin to. Must be less than `len(self.ks)`
        weighted : bool, optional
            Whether to do a weighted or unweighted average over bins. Default
            is `True`.
            
        Return
        ------
        pkmu : PkmuMeasurement
            A copy of `self` with the rebinned `data` attribute
        """  
        ks = np.asarray(self._data.index.levels[self._k_level], dtype=float)      
        if N >= len(ks):
            raise ValueError("Can only re-bin into fewer than %d k bins" %len(ks))
           
        # return a copy   
        toret = self.copy() 
        
        # compute the integer numbers for the new k bins
        dk = np.diff(ks)
        bins = np.linspace(np.amin(ks)-0.5*dk[0], np.amax(ks)+0.5*dk[-1], N+1)
        values = np.array([index[self._k_level] for index in self._data.index.get_values()])
        bin_numbers = np.digitize(values, bins) - 1
        
        # make it a series and add it to the data frame
        bin_numbers = Series(bin_numbers, name='bin_number', index=self._data.index)
        toret._data['bin_number'] = bin_numbers
        
        # group by bin number and compute mean k values in each bin
        bin_groups = toret._data.reset_index(level=self._k_level).groupby(['bin_number'])
           
        # replace "bin_number", conditioned on bin_number == new_kbins
        new_kbins = bin_groups.k.mean() 
        toret._data['k'] = new_kbins.loc[toret._data.bin_number].values
        del toret._data['bin_number']
        
        # group by the new k bins
        groups = toret._data.reset_index(level=self._mu_level).groupby(['mu', 'k'])

        # apply the average function, either do weighted or unweighted
        new_frame = groups.apply(tools.groupby_average, weighted)
        
        # delete unneeded columns
        del new_frame['k'], new_frame['mu']
        
        toret._data = new_frame
        return toret
    #end rebin_k
    
    #---------------------------------------------------------------------------
    def Pk_kaiser(self, bias):
        """
        Return the biased, Kaiser real-space power spectrum at the `k` 
        values defined for this measurement. This is independent of `mu`.
        The returned expression is 
        
        :math: P_{kaiser}(k) = bias^2 * P_{linear}(k)
        
        Note: this returns the Eisenstein-Hu no-wiggle power spectrum
        
        Parameters
        ----------
        bias : float
            The linear bias factor
            
        Returns
        -------
        Pk_kaiser : np.ndarray
            The real-space Kaiser P(k) in units specified by `self.output_power_units`,
            and defined at the `k` values of `self.ks`
        """
        if self.cosmo is None or self.redshift is None:
            raise ValueError("Redshift and cosmology must be defined for `Pk_kaiser`.")
            
        # conversion factor for output units, as specified in `self.output_units`
        power_units_factor = self._h_conversion_factor('power', 'relative', self.output_power_units)
        
        # try not to recompute the linear power if nothing changed
        try:
            return (bias**2 * self._Pk_lin) * power_units_factor
        except AttributeError:            
            # linear power spectrum is in "relative" units, so convert wavenumber
            # units appropriately
            factor = self._h_conversion_factor('wavenumber', self.output_k_units, 'relative')
            ks = self.ks * factor
            
            # set the transfer function to EH no-wiggle
            self.cosmo.SetTransferFunction(pygcl.Cosmology.EH_NoWiggle)
            power = pygcl.LinearPS(self.cosmo, self.redshift)
            self._Pk_lin = power(ks)
            
            # return the biased power with correct units
            return (bias**2 * self._Pk_lin) * power_units_factor 
    #end Pk_kaiser
    
    #---------------------------------------------------------------------------
    def save(self, filename, use_pickle=True):
        """
        Save the `PkmuMeasurements` instance as a pickle to the filename specified
        
        Parameters
        ----------
        filename : str 
            the filename to output to
        """   
        if use_pickle:     
            pickle.dump(self, open(filename, 'w'))
        else:
            store = HDFStore(filename, 'w')
            store['data'] = self._data
            self._data = None
            store.get_storer('data').attrs.power = self
            store.close()
    #end save

#endclass PowerMeasurement
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class PkmuMeasurement(PowerMeasurement):
    """
    A subclass of `PowerMeasurement` designed to hold a P(k, mu) measurement
    """
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
        if isinstance(k, int): k = self._convert_k_integer(k)
        if isinstance(mu, int): mu = self._convert_mu_integer(mu)

        if not (mu, k) in self.data.index:
            raise KeyError("Sorry, band (k = %s, mu = %s) does not exist" %(k, mu))

        return self.data.loc[mu, k]
    #end __getitem__
    
    #---------------------------------------------------------------------------
    def _convert_mu_integer(self, mu_int):
        """
        Convert the integer value specifying a given `mu` to the actual value
        """
        msg = "[0, %d) or [-%d, -1]" %(len(self.mus), len(self.mus))
        if mu_int < 0: mu_int += len(self.mus)
        if not 0 <= mu_int < len(self.mus):
            raise KeyError("Integer that identifies mu-band must be between %s" %msg)
        return self.mus[mu_int]
    #end _convert_mu_integer

    #---------------------------------------------------------------------------
    @property
    def data(self):
        """
        The `pandas.DataFrame` holding the power measurement, in units
        specified by `self.output_power_units`
        """
        factor = self._h_conversion_factor('power', self.measurement_units['power'], self.output_power_units)
        
        # normalize possibly        
        if self.normalize:
            if not hasattr(self, 'bias'):
                raise AttributeError("Must define `bias` attribute if you wish to have normalized power output")

            norms = np.array([self.normalization(self.bias, mu=mu) for mu in self.mus])
            norm = Series(norms.flatten(), index=self._data.index)
            factor /= norm
            
        # subtract the shot noise possibly
        toret = self._data.copy()
        cols = ['power', 'baseline', 'noise', 'shot_noise']
        if self.subtract_shot_noise:
            toret[cols] = toret[cols].subtract(toret['shot_noise'], axis='index')
        
        # multiply the factor and return
        toret[cols] = toret[cols].multiply(factor, axis='index')
        toret['variance'] = toret['variance'].multiply(factor**2, axis='index')
        
        return toret
    #end data
    
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
         if "k" not in self._data.index.names:
             raise ValueError("Index for `self.data` does not contain `k`")
         index = np.where(np.asarray(self._data.index.names) == 'k')[0]
         self.__k_level = index[0]
         return self.__k_level
    #end _k_level  
     
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
            if "mu" not in self._data.index.names:
                raise ValueError("Index for P(k, mu) frame does not contain 'mu'")
            index = np.where(np.asarray(self._data.index.names) == 'mu')[0]
            self.__mu_level = index[0]
            return self.__mu_level
    #end _mu_level
    
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
        return np.asarray(self._data.index.levels[self._mu_level], dtype=float)
    #end mus
    
    #---------------------------------------------------------------------------
    # the functions        
    #---------------------------------------------------------------------------
    def rebin_mu(self, N, weighted=True):
        """
        Return a copy of `self` with the `data` DataFrame rebinnned, with
        a total of `N` `mu` bins
        
        Parameters
        ----------
        N : int
            The number of mu bins to rebin to. Must be less than `len(self.mus)`
        weighted : bool, optional
            Whether to do a weighted or unweighted average over bins. Default
            is `True`.
            
        Return
        ------
        pkmu : PkmuMeasurement
            A copy of `self` with the rebinned `data` attribute
        """        
        if N >= len(self.mus):
            raise ValueError("Can only re-bin into fewer than %d mu bins" %len(self.mus))
           
        # return a copy   
        toret = self.copy() 
        
        # first compute the new mu bins
        dmu = 1./N
        new_bins = np.array([index[self._mu_level] for index in self._data.index.get_values()]) // dmu
        new_bins = 0.5*dmu + new_bins*dmu
        
        # make it a series and add it to the data frame
        new_bins = Series(new_bins, name='mu', index=self._data.index)
        toret._data['mu'] = new_bins
        
        # make groups
        groups = toret._data.reset_index(level=self._k_level).groupby(['mu', 'k'])

        # apply the average function, either do weighted or unweighted
        new_frame = groups.apply(tools.groupby_average, weighted)
        
        # delete unneeded columns
        del new_frame['k'], new_frame['mu']
        
        toret._data = new_frame
        return toret
    #end rebin_mu    
            
    #---------------------------------------------------------------------------
    def Pk(self, mu):
        """
        Return the power measured P(k) at a specific value of mu, as a 
        `pandas.DataFrame`.  The units of the output are specified by 
        `self.output_power_units`
        
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
        # get the correct value of mu, if an integer was specified
        if isinstance(mu, int): mu = self._convert_mu_integer(mu)
            
        # make sure the power is defined at this value of mu
        if mu not in self.mus:
            raise ValueError("Power measurement not defined at mu = %s" %mu)
        
        # return P(k)
        toret = self.data.xs(mu, level=self._mu_level)
        toret.ks = self.ks
        return toret
    #end Pk
    
    #---------------------------------------------------------------------------
    def Pmu(self, k):
        """
        Return the power measured P(mu) at a specific value of k.  The units of 
        the output are specified by `self.output_power_units`
        
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
        # get the correct value of k, if an integer was specified
        if isinstance(k, int): k = self._convert_k_integer(k)
            
        # make sure the power is defined at this value of k
        if k not in self.ks:
            raise ValueError("Power measurement not defined at k = %s" %k)
        
        # return P(mu)
        toret = self.data.xs(k, level=self._k_level)
        toret.mus = self.mus
        return toret
    #end Pmu
    
    #---------------------------------------------------------------------------
    def mu_averaged(self, weighted=True):
        """
        Return the mu averaged power measurement, as a ``pandas.DataFrame``
        object, optionally computing the weighted average of the power for a 
        given ``mu`` value. The units of the output are specified by 
        `self.output_power_units`
        
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
        # group by k
        grouped = self.data.groupby(level=['k'])
        avg_data = grouped.apply(tools.groupby_average, weighted)            
        avg_data.ks = self.ks
        
        return avg_data
    #end mu_averaged
    
    #---------------------------------------------------------------------------
    def Pkmu_kaiser(self, mu, bias):
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
            
        Returns
        -------
        Pkmu_kaiser : np.ndarray
            The redshift-space Kaiser P(k, mu) in units specified by 
            `self.output_power_units` and defined at the `k` values of `self.ks`
        """
        beta = self.f / bias
        return (1. + beta*mu**2)**2 * self.Pk_kaiser(bias)
    
    #end Pkmu_kaiser
            
    #---------------------------------------------------------------------------
    def mu_averaged_Pkmu_kaiser(self, bias):
        """
        Return the mu-averaged, biased, Kaiser redshift-space power spectrum 
        at the `k` values defined for this measurement. This function 
        takes the average of the output of `self.Pkmu_kaiser` for each `mu`
        value defined in `self.mus` 
        
        Parameters
        ----------
        bias : float
            The linear bias factor
            
        Returns
        -------
        Pk : np.ndarray
            The mu-averaged, Kaiser redshift space spectrum in units specified 
            by `self.output_powe_units`, and defined at the `k` values of `self.ks`
        """
        return np.mean([self.Pkmu_kaiser(mu, bias) for mu in self.mus], axis=0)

    #end mu_averaged_Pkmu_kaiser
    
    #---------------------------------------------------------------------------
    def normalization(self, bias, mu=None, mu_avg=False):
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
        """
        if self.space == 'real':
            Plin_norm = self.Pk_kaiser(bias)
        elif self.space == 'redshift':
            if mu_avg:
                Plin_norm = self.mu_averaged_Pkmu_kaiser(bias)
            else:
                Plin_norm = self.Pkmu_kaiser(mu, bias)
        else:
            raise ValueError("Attribute `space` must be defined for this function.")
        
        return Plin_norm
    #end normalization
            
    #---------------------------------------------------------------------------
    def plot(self, *args, **kwargs):
        """
        Plot either the P(k, mu) measurement in the units specified by 
        `self.output_power_units`.  In addition to the keywords accepted below, 
        any plotting keywords accepted by `matplotlib` can be passed.

        Parameters
        ----------
        ax : plotify.Axes, optional
            This can be specified as the first positional argument, or as the
            keyword `ax`. If not specified, the plot will be added to the 
            axes returned by `plotify.gca()`
            
            
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
            
        subtract_constant_noise : bool, optional
            If `True`, subtract the 1/nbar shot noise stored in `shot_noise`
            column of `self.data` before plotting. Default is `False`. 

        subtract_noise : bool, optional
            If `True`, subtract the noise component stored as the `noise`
            column in `self.data`. Default is `False`. 

        normalize_by_baseline : bool, optional
            Normalize by the baseline, plotting data/baseline. This is only
            possible for the `Pkmu` `data_type`. Default is `False`
            
        show_baseline : bool, optional
            Plot the baseline over the data points as a solid black line. 
            Default is `False`.
            
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
        label              = kwargs.pop('label', None)
        y_offset           = kwargs.pop('offset', 0.)
        weighted_mean      = kwargs.pop('weighted_mean', True)
        add_axis_labels    = kwargs.pop('add_axis_labels', True)
        normalize          = kwargs.pop('normalize_by_baseline', False)
        show_baseline      = kwargs.pop('show_baseline', False)

        # check if we need to extract a bias value
        if isinstance(bias, basestring):
            assert exists(bias), "Specified bias TSAL file does not exist"
            bias, bias_err = tools.extract_bias(bias)
            
        # check that both plot_linear and norm_linear are not True
        assert not (plot_linear == True and norm_linear == True), \
            "Boolean keywords `norm_linear` and `plot_linear` must take different values."
                   
        # now get the data
        k = self.ks 
        norm = 1.
        # mu-averaged or not
        mu = kwargs.pop('mu', None)
        mu_avg = mu is None
        if mu_avg:
            mu   = np.mean(self.mus)
            data = self.mu_averaged(weighted=weighted_mean)
        else:
            # get the mu value, correctly 
            if isinstance(mu, int): mu = self._convert_mu_integer(mu)
            data = self.Pk(mu)

        Pk   = data.power
        err  = data.variance**0.5
        if normalize: norm = data.baseline
        
        # the shot noise
        noise = np.zeros(len(Pk))
        if sub_noise: noise = data.noise
        elif sub_constant_noise: noise = data.shot_noise
          
        # plot the linear theory result
        if plot_linear or norm_linear:

            # the normalization
            norm = self.normalization(bias, mu=mu, mu_avg=mu_avg)
    
            # plot both the linear normalization and result separately
            if plot_linear:
                
                # set up labels and plot the linear normalization
                lin_label = r'$P^\mathrm{EH}(k, \mu)$'
                P_label = r"$P(k, \mu = %.3f)$ "%mu
  
                pfy.loglog(ax, k, norm, c='k', label=lin_label)
                pfy.errorbar(ax, k, Pk-noise, err, label=P_label, **kwargs)
            
            # normalize by the linear theory
            else:
                if label is None: label = r"$P(k, \mu = %.3f)$ "%mu
                data_to_plot = (k, (Pk-noise)/norm, err/norm)
                pfy.plot_data(ax, data=data_to_plot, labels=label, y_offset=y_offset, plot_kwargs=kwargs)
        
        # just plot the measurement with no normalization
        else:
            
            if label is None: label = r"$P(k, \mu = %.3f)$ "%mu
            pfy.errorbar(ax, k, (Pk - noise)/norm, err/norm, label=label, **kwargs)

        if show_baseline: 
            pfy.plot(ax, k, (data.baseline-noise)/norm, c='k')
        
        # now determine the units
        if self.output_power_units == 'relative':
            P_units = "(Mpc/h)^3"
        else:
            P_units = "(Mpc)^3"
            
        if self.output_k_units == 'relative':
            k_units = "(h/Mpc)"
        else:
            k_units = "(1/Mpc)"
      
        if add_axis_labels:
            # let's set the xlabels and return
            if ax.xlabel.text == "":
                ax.xlabel.update(r"$\mathrm{k \ %s}$" %k_units)
            
            if ax.ylabel.text == "":
                P_label = r"P(k, \mu)"
                if not normalize:
                    norm_label = r"P^\mathrm{EH}(k, \mu)"
                else:
                    norm_label = r"P_\mathrm{base}(k, \mu)"

                if mu_avg: 
                    P_label = r"\langle %s \rangle_\mu" %P_label
                    norm_label = r"\langle %s \rangle_\mu" %norm_label

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
    def write(self, filename, columns=True):
        """
        Save the power measurement to a data file in ASCII format

        Parameters
        ----------
        filename : str
            The name of the file to write to
        """ 
        k_units = "h/Mpc" if self.output_k_units == 'relative' else "1/Mpc"
        power_units = "(Mpc/h)^3" if self.output_power_units == 'relative' else "(Mpc)^3"

        shot_noise_hdr = ""
        if self.subtract_shot_noise:
            shot_noise_hdr = "shot noise subtracted: P_shot = %.5e %s\n" %(self.shot_noise, power_units)
        header = "P(k, mu) in %s space at redshift z = %s in units of %s\n" %(self.space, self.redshift, power_units) + \
                 "for k = %.5f to %.5f %s,\n" %(np.amin(self.ks), np.amax(self.ks), k_units) + \
                 "number of wavenumbers equal to %d,\n" %(len(self.ks)) + \
                 "number of mu bins equal to %d\n" %(len(self.mus)) + \
                 shot_noise_hdr
           
        # determine if we have modes
        has_modes = hasattr(self.data, 'modes')      
                 
        if not columns: 
            header += "{x}1:mu {x}2:k_cen {x}3:power {x}4:error".format(x=" "*5)
                        
            if has_modes: 
                header += (" "*5 + "modes")
            if self.subtract_shot_noise:
                header += (" "*5 + "shot_noise")
        else:
            tmp = "%s1:k_cen (%s) " %(" "*5, k_units)
            tmp += (" ").join(["%d:P_mu=%s %d:err_mu=%s" %(2*(i+1), mu, 2*i+3, mu) for i, mu in enumerate(self.mus)])
            
            if has_modes: 
                tmp += " modes"
            if self.subtract_shot_noise:
                tmp += " shot_noise"
            header += tmp 
                   
            
        if not columns:
            ks, mus, power, error, modes, shot_noise = [], [], [], [], [], []
            N_ks = len(self.ks)
            for mu in self.mus:
                mus += [mu]*N_ks
                Pk = self.Pk(mu)
                ks += list(self.ks)
                power += list(Pk.power)
                error += list(Pk.variance**0.5)
                if has_modes:
                    modes += list(Pk.modes)
                if self.subtract_shot_noise:
                    shot_noise += [self.shot_noise]*N_ks
                    
            data = (mus, ks, power, error)
            if has_modes:
                data += (modes,)
            if self.subtract_shot_noise:
                data += (shot_noise,)
            
        else:
            data = [self.ks]
            for mu in self.mus:
                Pk = self.Pk(mu)
                data.append(Pk.power)
                data.append(Pk.variance**0.5)
                if has_modes:
                    data.append(Pk.modes)
                if self.subtract_shot_noise:
                    data.append(np.ones(len(self.ks))*self.shot_noise)
                
        toret = np.vstack(data).T
        np.savetxt(filename, toret, header=header, fmt="%20.5e")
    #end write

#endclass PkmuMeasurement

#-------------------------------------------------------------------------------
class PoleMeasurement(PowerMeasurement):
    """
    A subclass of `PowerMeasurement` designed to hold a multipole measurement
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize and keep track of what order the multipole is
        """
        self._order = kwargs.pop('order', 0)
        assert self._order in [0, 2, 4], "Can only compute multipoles l = 0, 2, 4"
        
        if isinstance(args[0], basestring):
            assert len(args) == 2, "Must specify two arguments to `PoleMeasurement`"
            super(PoleMeasurement, self).__init__(*args, **kwargs)
        elif isinstance(args[0], PkmuMeasurement):
            units, kwargs = self._data_from_pkmu(args[0])
            if len(args) == 2 and units != args[1]:
                raise ValueError("Units mismatch in measuring multipoles from pkmu")
            super(PoleMeasurement, self).__init__(None, units, **kwargs)
            
        # set shot noise to zero for quadrupole
        if (self._order > 0):
            self._data.shot_noise = 0.
        
    #end __init__
    
    #---------------------------------------------------------------------------
    def _fill_missing_data(self, pkmu):
        """
        Fill missing data by interpolating a spline
        """
        from scipy.interpolate import UnivariateSpline as spline
        
        # get the return frame
        toret = pkmu.data
        
        # loop over each k bin
        ks = toret.index.get_level_values('k').unique()
        for i, k in enumerate(ks):
            
            x = toret.xs(k, level='k', drop_level=False)
            inds = x.power.isnull()
            y = x.dropna()
            
            # now fill with spline values
            if inds.sum() > 0:
                s = spline(y.index.get_level_values('mu'), y.power, w=1./y.variance**0.5)
                mus = x.index.get_level_values('mu')[inds]
                toret.loc[x.index[inds], 'power'] = s(mus)
            
        return toret
            
    #---------------------------------------------------------------------------
    def _data_from_pkmu(self, pkmu):
        """
        Internal function to compute the multipole from a PkmuMeasurement object
        """
        from scipy.special import legendre
        from scipy.integrate import quad
        
        # fill the missing data
        data = self._fill_missing_data(pkmu)
        
        # initialize blank arrays
        power, variance = pkmu.ks*0., pkmu.ks*0.
        
        # loop over all mus
        mus = data.index.get_level_values('mu').unique()
        Nmu = len(mus)
        dmu = np.diff(mus)[0]
        for i, mu in enumerate(mus):   
            
            # the integral over the legendre polynomial at this mu
            mu_integral = quad(lambda x: legendre(self._order)(x), mu-0.5*dmu, mu+0.5*dmu)[0]
            
            # P(k) at this mu
            Pk_thismu = data.xs(mu, level='mu')
            
            # do a weighted sum, accounting for differences in mus
            power += mu_integral* np.nan_to_num(Pk_thismu.power)
            variance += mu_integral**2 * np.nan_to_num(Pk_thismu.variance)
        
        # normalize properly
        power *= (2*self._order + 1)
        variance *= (2*self._order + 1)**2
        
        # now make the dataframe
        index = Index(pkmu.ks, name='k')
        self._data = DataFrame(data=np.vstack((power, variance)).T, index=index, columns=['power', 'variance'])   
    
        # get the kwargs
        keys = ['cosmo', 'space', 'redshift']
        kwargs = {}
        for k in keys:
            if hasattr(pkmu, k): kwargs[k] = getattr(pkmu, k)
        kwargs['Pshot'] = pkmu.shot_noise if not pkmu.subtract_shot_noise else 0.
        
        return pkmu.measurement_units['k'], kwargs
    #end _data_from_pkmu
        
    #---------------------------------------------------------------------------
    def _data_from_tsal(self, tsal_file):
        """
        Internal function to read and store a power multipole measurement 
        from a TSAL file
        """
        ff = open(tsal_file)
        self.readTSAL(ff)
        ff.close()
        
        pole = {}
        tags = {0 : 'mono', 2 : 'quad', 4 : 'hexadec', 6 : 'tetrahexadec'}
        for key, val in self.pars.iteritems():

            k = float(key.split('_')[-1])
            if tags[self.order] in key:
                pole[k] = (val.val, val.err)
                
        if len(pole) > 0:
            # set up the index for multipole data frames 
            ks = np.array(sorted(pole.keys()))
            index = Index(ks, name='k')    

            # now make the data frame
            vals, errs = map(np.array, zip(*[pole[k] for k in ks]))
            self._data = DataFrame(data=np.vstack((vals, errs**2)).T, index=index, columns=['power', 'variance'])
            
            # remove power if it's zero
            to_replace = np.isclose(self._data.power, 0.)
            self._data.loc[to_replace, ['power', 'variance']] = np.NaN
            
        else:
            self._data = None

    #end _data_from_tsal
    
    #---------------------------------------------------------------------------
    def __getitem__(self, k):
        """
        Access individual band-powers through dict-like interface:

            Pkmu_frame = self[k], 

        where k is either an integer between 0 - len(self.ks) 
        or a floats specifying the actual value

        Returns
        -------
        frame : pandas.DataFrame
            A DataFrame with a single row holding the data for the band, 
            which columns specified by `self.columns`
        """
        if isinstance(k, int): k = self._convert_k_integer(k)

        if not k in self.data.index:
            raise KeyError("Sorry, band (k = %s) does not exist" %k)

        return self.data.loc[k]
    #end __getitem__
    
    #---------------------------------------------------------------------------
    @property
    def order(self):
        """
        Read-only variable specifying the `order` of the multipole
        """
        return self._order
       
    #---------------------------------------------------------------------------
    @property
    def data(self):
        """
        The `pandas.DataFrame` holding the power measurement, in units
        specified by `self.output_power_units`
        """
        factor = self._h_conversion_factor('power', self.measurement_units['power'], self.output_power_units)
        
        # normalize possibly        
        if self.normalize:
            if not hasattr(self, 'bias'):
                raise AttributeError("Must define `bias` attribute if you wish to have normalized power output")

            norm = Series(self.normalization(self.bias), index=self._data.index)
            factor /= norm
            
        # subtract the shot noise possibly
        toret = self._data.copy()
        if self.subtract_shot_noise: 
            cols = ['power', 'shot_noise']
            toret[cols] = toret[cols].subtract(toret['shot_noise'], axis='index')
        
        # multiply the factor and return
        toret = toret.multiply(factor, axis='index')
        toret['variance'] = toret['variance'].multiply(factor, axis='index')
        return toret
    #end data
    
    #---------------------------------------------------------------------------
    def normalization(self, bias):
        """
        Return the proper Kaiser normalization for this power measurement, 
        mostly used for plotting purposes.
        
        Parameters
        ----------
        bias : float
            The linear bias factor
        """
        if self.space == 'real':
            return self.Pk_kaiser(bias) 
        else:
            
            beta = self.f / bias
            if self.order == 0:
                return (1. + 2./3.*beta + 1./5.*beta**2) * self.Pk_kaiser(bias) 
            elif self.order == 2:
                return (4./3.*beta + 4./7.*beta**2) * self.Pk_kaiser(bias)
            else:
                return (8./35.*beta**2) * self.Pk_kaiser(bias)
            
    #end normalization
            
    #---------------------------------------------------------------------------
    def plot(self, *args, **kwargs):
        """
        Plot the multipole measurement, in the units specified by 
        `self.output_power_units`.  In addition to the keywords accepted below, any 
        plotting keywords accepted by `matplotlib` can be passed.

        Parameters
        ----------
        ax : plotify.Axes, optional
            This can be specified as the first positional argument, or as the
            keyword `ax`. If not specified, the plot will be added to the 
            axes returned by `plotify.gca()`
            
        bias : {float, str}, optional
            If a `float`, the bias of the sample. If a `str`, the name of file
            that contains the best-fit linear bias value. Default is `1`.
            
        norm_linear : bool, optional
            Plot the power spectrum normalized by the linear spectrum. Default
            is `False`.
            
        plot_linear : bool, optional
            Plot the linear spectrum as well as the data spectrum. Default
            is `False`.
            
        subtract_constant_noise : bool, optional
            If `True`, subtract the 1/nbar shot noise stored in `shot_noise`
            column of `self.data before plotting. Default is `False`. 
            
        label : str, optional
            The label to attach to these plot lines. Default is `None`
        
        offset : float, optional
            Offset the plot in the y-direction by this amount. Default is `0`
            
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
        bias               = kwargs.pop('bias', 1.)
        norm_linear        = kwargs.pop('norm_linear', False)
        plot_linear        = kwargs.pop('plot_linear', False)
        label              = kwargs.pop('label', None)
        y_offset           = kwargs.pop('offset', 0.)
        add_axis_labels    = kwargs.pop('add_axis_labels', True)

        # check if we need to extract a bias value
        if isinstance(bias, basestring):
            assert exists(bias), "Specified bias TSAL file does not exist"
            bias, bias_err = tools.extract_bias(bias)
            
        # check that both plot_linear and norm_linear are not True
        assert not (plot_linear == True and norm_linear == True), \
            "Boolean keywords `norm_linear` and `plot_linear` must take different values."
            
        # now get the data
        k = self.ks 
        Pk   = self.data.power
        err  = self.data.variance**0.5

        # the shot noise
        noise = np.zeros(len(Pk))
        if sub_constant_noise: noise = self.data.shot_noise
          
        # plot the linear theory result
        if plot_linear or norm_linear:

            # the normalization
            norm = self.normalization(bias)

            # plot both the linear normalization and result separately
            if plot_linear:
                
                # set up labels and plot the linear normalization
                lin_label = r'$P^\mathrm{EH}_{\ell=%s}(k)$' %(self.order)
                P_label = r"$P_{\ell=%s}(k)$" %(self.order)
  
                pfy.loglog(ax, k, norm, c='k', label=lin_label)
                pfy.errorbar(ax, k, Pk-noise, err, label=P_label, **kwargs)
            
            # normalize by the linear theory
            else:
                data_to_plot = (k, (Pk-noise)/norm, err/norm)
                pfy.plot_data(ax, data=data_to_plot, labels=label, y_offset=y_offset, plot_kwargs=kwargs)
        
        # just plot the measurement with no normalization
        else:
            if label is None: label = r"$P_{\ell=%s}(k)$" %(self.order)
            pfy.errorbar(ax, k, (Pk - noise), err, label=label, **kwargs)
            
        # now determine the units
        if self.output_power_units == 'relative':
            P_units = "(Mpc/h)^3"
        else:
            P_units = "(Mpc)^3"
            
        if self.output_k_units == 'relative':
            k_units = "(h/Mpc)"
        else:
            k_units = "(1/Mpc)"
        
      
        if add_axis_labels:
            # let's set the xlabels and return
            if ax.xlabel.text == "":
                ax.xlabel.update(r"$\mathrm{k \ %s}$" %k_units)
            
            if ax.ylabel.text == "":
                norm_label = r"P^\mathrm{EH}_{\ell=%s}(k)" %(self.order)
                P_label = r"P_{\ell=%s}(k)" %(self.order)
            
                if norm_linear:
                    if sub_constant_noise:
                        ax.ylabel.update(r"$\mathrm{(%s - \bar{n}^{-1}) \ / \ %s}$" %(P_label, norm_label))
                    else:
                        ax.ylabel.update(r"$\mathrm{%s \ / \ %s}$" %(P_label, norm_label))
            
                else: 
                    if sub_constant_noise:
                        ax.ylabel.update(r"$\mathrm{%s - \bar{n}^{-1} \ %s}$" %(P_label, P_units))
                    else:       
                        ax.ylabel.update(r"$\mathrm{%s \ %s}$" %(P_label, P_units))

        return ax.get_figure(), ax
    
    #end plot   
    
    #---------------------------------------------------------------------------
    def write(self, filename):
        """
        Save the power multipole measurement to a data file in ASCII format
        """
        if (self._order == 0):
            name = "Monopole"
        elif (self._order == 2):
            name = "Quadrupole"
        else:
            name = "Hexadecapole"
            
        k_units = "h/Mpc" if self.output_k_units == 'relative' else "1/Mpc"
        power_units = "(Mpc/h)^3" if self.output_power_units == 'relative' else "(Mpc)^3"
        
        header = "%s P_{\ell}(k) in %s space at redshift z = %s\n" %(name, self.space, self.redshift) + \
                 "for k_cen = %.5f to %.5f %s,\n" %(np.amin(self.ks), np.amax(self.ks), k_units) + \
                 "number of wavenumbers equal to %d\n" %(len(self.ks)) + \
                 "%s1:k (%s)%s2:P %s%s3:error %s" %(" "*5, k_units, " "*10, power_units, " "*8, power_units)
        
        toret = np.vstack((self.ks, self.data.power, self.data.variance**0.5)).T
        np.savetxt(filename, toret, header=header, fmt="%20.5e")
    #end write
    
    #---------------------------------------------------------------------------
#endclass PoleMeasurement

#-------------------------------------------------------------------------------
class MonopoleMeasurement(PoleMeasurement):
    """
    A subclass of `PoleMeasurement` designed to hold a monopole measurement
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize and set the `order` to `0`
        """
        kwargs['order'] = 0
        PoleMeasurement.__init__(self, *args, **kwargs)
        
    #end __init__
    
#-------------------------------------------------------------------------------
class QuadrupoleMeasurement(PoleMeasurement):
    """
    A subclass of `PoleMeasurement` designed to hold a quadrupole measurement
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize and set the `order` to `2`
        """
        kwargs['order'] = 2
        PoleMeasurement.__init__(self, *args, **kwargs)
    #end __init__
    
#-------------------------------------------------------------------------------
class HexadecapoleMeasurement(PoleMeasurement):
    """
    A subclass of `PoleMeasurement` designed to hold a hexadecapole measurement
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize and set the `order` to `4`
        """
        kwargs['order'] = 4
        PoleMeasurement.__init__(self, *args, **kwargs)
    #end __init__
    
#-------------------------------------------------------------------------------
class PowerMeasurements(dict):
    """
    A dict-like object to hold a series of `PowerMeasurement` objects
    """
    def __init__(self, cpm_tsal, units, poles_tsal=None, **kwargs):
        
        # the P(k,mu) measurement
        self['pkmu'] = PkmuMeasurement(cpm_tsal, units, **kwargs)
        
        # do the multipoles too
        if poles_tsal is not None:
            self['monopole']     = MonopoleMeasurement(poles_tsal, units, **kwargs)
            self['quadrupole']   = QuadrupoleMeasurement(poles_tsal, units, **kwargs)
            
            # try to read the hexadecapole
            try:
                self['hexadecapole'] = HexadecapoleMeasurement(poles_tsal, units, **kwargs)
            except:
                self['hexadecapole'] = None                        
    #end __init__
    
    #---------------------------------------------------------------------------
    @property
    def pkmu(self):
        return self['pkmu']
        
    @property
    def monopole(self):
        return self['monopole']
        
    @property
    def quadrupole(self):
        return self['quadrupole']
    
    @property
    def hexadecapole(self):
        return self['hexadecapole']
        
    #---------------------------------------------------------------------------
    def save(self, filename):
        """
        Save the `PkmuMeasurements` instance as a pickle to the filename specified
        
        Parameters
        ----------
        filename : str 
            the filename to output to
        """        
        pickle.dump(self, open(filename, 'w'))
    #end save
    
    #---------------------------------------------------------------------------
#endclass PowerMeasurements

#-------------------------------------------------------------------------------    
        
    
