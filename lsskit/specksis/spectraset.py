import xarray as xr

from . import utils, tools
from .. import numpy as np

class SpectraSet(xr.DataArray):
    """
    N-dimensional set of power spectra measurements
    """

    def __init__(self, data, coords=None, dims=None, name=None,
                 attrs=None, **kwargs):
        """
        Parameters
        ----------
        data : array_like
            Values for this array. Must be an ``numpy.ndarray``, ndarray like,
            or castable to an ``ndarray``.
        coords : sequence or dict of array_like objects, optional
            Coordinates (tick labels) to use for indexing along each dimension.
            If dict-like, should be a mapping from dimension names to the
            corresponding coordinates.
        dims : str or sequence of str, optional
            Name(s) of the the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            are taken from ``coords`` (if possible) and otherwise default to
            ``['dim_0', ... 'dim_n']``.
        name : str or None, optional
            Name of this array.
        attrs : dict_like or None, optional
            Attributes to assign to the new variable. By default, an empty
            attribute dictionary is initialized.
        """
        super(SpectraSet, self).__init__(data, coords=coords, dims=dims, name=name, attrs=attrs, **kwargs)
        
 
    @classmethod
    def from_files(cls, loader, result_dir, basename, coords, dims=None, ignore_missing=False, args=(), kwargs={}):
        """
        Return a SpectraSet instance by loading data from all files in 
        ``result_dir`` with base ``basename``. The filename is formatting using 
        the specified coordinates and dimensions
        
        Parameters
        ----------
        loader : callable   
            the function to call to load the data -- must take the filename
            as first argument
        result_dir : str
            the directory holding the files to load the spectra from
        basename : str
            a string specifying the basename of the files, with `{}` 
            occurences where the string will be formatted according
            to the dimension names and coordinate values
        coords : sequence or dict of array_like objects, optional
            Coordinates to use for indexing along each dimension.
            If dict-like, should be a mapping from dimension names to the
            corresponding coordinates.
        dims : str or sequence of str, optional
            Name(s) of the the data dimension(s). Must be either a string (only
            for 1D data) or a sequence of strings with length equal to the
            number of dimensions. If this argument is omitted, dimension names
            are taken from ``coords``, which must be dict-like
        args : tuple, optional
            additional arguments to pass to `loader` after the filename
        kwargs : dict, optional
            additional keywords to pass to `loader`
        """
        if dims is None:
            if not utils.is_dict_like(coords):
                raise TypeError("if no `dims` provided, `coords` must be a dict")
            dims = coords.keys()
            coords = coords.values()
        if len(dims) != len(coords):
            raise ValueError("shape mismatch between supplied `dims` and `coords`")
            
        data = np.empty(map(len, coords), dtype=object)
        for i, f in utils.enum_files(result_dir, basename, dims, coords, ignore_missing=ignore_missing):
            try:
                data[i] = loader(f, *args, **kwargs)
            except Exception as e:
                if ignore_missing:
                    data[i] = np.nan
                else:
                    raise Exception(e)
        return SpectraSet(data, coords=coords, dims=dims)


    def ndindex(self, dims=None):
        """
        A generator to iterate over the specified dimensions, yielding a
        dictionary holding the index keys
        
        Parameters
        ----------
        dims : list or basestring, optional
            A list or single string specifying the dimension names to 
            iterate over
        """
        if dims is None:
            dims = self.dims
        if isinstance(dims, basestring):
            dims = [dims]
        
        if not len(dims):
            key = {k:v.values.tolist() for k,v in self.coords.items()}
            yield key
        else:
            for d in utils.ndindex(dims, self.coords):
                val = self.loc[d]                   
                key = {k:v.values.tolist() for k,v in val.coords.items()}
                yield key
                
    def nditer(self, dims=None):
        """
        A generator to iterate over the specified dimensions, yielding a
        dictionary holding the index keys/values and the slice of the
        SpectraSet
        
        Parameters
        ----------
        dims : list or basestring
            A list or single string specifying the dimension names to 
            iterate over
        """
        if dims is None:
            dims = self.dims
        if isinstance(dims, basestring):
            dims = [dims]
        
        if not len(dims):
            key = {k:v.values for k,v in self.coords.items()}
            yield key, self
        else:
            for d in utils.ndindex(dims, self.coords):
                val = self.loc[d]
                key = {k:v.values for k,v in val.coords.items()}
                yield key, val

            
    def add_corr_errors(self):
        """
        Add correlation function errors to each object in the set
        """
        for coord, corr in self.nditer():
            corr = corr.values
            if 'error' in corr: continue
            DD = (corr['corr']+1)*corr['RR']
            err = (corr['corr']+1)/DD**0.5
            corr['error'] = err
            
    def add_power_errors(self, power_x1=None, power_x2=None):
        """
        Add power spectrum errors to each object in the set
        
        Parameters
        ----------
        power_x1, power_x2 : `Power1dDataSet`, `Power2dDataSet`, optional
            If the set stores a cross-power measurement, the auto power
            measurements are needed to compute the error
        """
        p1 = p2 = None
        for coord, power in self.nditer():
            if power.isnull(): continue
            
            power = power.values
            if 'error' in power: continue
            if power_x1 is not None:
                coord1 = {k:coord[k] for k in power_x1.dims}
                p1 = power_x1.loc[coord1].values
            if power_x2 is not None:
                coord2 = {k:coord[k] for k in power_x2.dims}
                p2 = power_x2.loc[coord2].values
            utils.add_power_errors(power, p1, p2)
            
    def add_power_pole_errors(self, pkmu):
        """
        Add power spectrum multipole errors to each object in the set
        
        Parameters
        ----------
        pkmu : `Power2dDataSet`
            the P(k,mu) spectra set from which the multipole
            errors will be computed
        """
        for key in self.ndindex():
            if 'ell' not in key:
                raise ValueError("trying to add multipole errors without an `ell` dimension")
            this_pole = self.sel(**key).values
            ell = key.pop('ell')
            if len(key):
                this_pkmu = pkmu.sel(**key).values
            else:
                this_pkmu = pkmu
            utils.add_power_pole_errors(this_pole, this_pkmu, ell)
            
    def average(self, weights=None, axis=None):
        """
        Compute the average of SpectraSet, optionally weighting, over
        the axis specified 
        """
        if axis is None:
            axis = list(self.dims)
        else:
            if isinstance(axis, int):
                axis = [self.dims[axis]]
            elif isinstance(axis, str):
                axis = [axis]
            if not all(k in self.dims for k in axis):
                raise ValueError("specified `axis` must be one of %s" %str(self.dims))
                    
        # the number of measurements to average over
        N = np.prod([len(self[k]) for k in axis])
        first = {k:0 for k in axis}
        first = (self.isel(**first))
        
        scalar = sorted(axis) == sorted(self.dims)
                
        def compute_mean(first_, extra_key={}):    
            sum_only_ = []
            if hasattr(first_, '_fields_to_sum'):
                sum_only_ = first_._fields_to_sum
        
            # the weights
            if weights is None:
                weights_ = np.ones((N,) + first_.shape)
            else:
                if isinstance(weights, str):
                    if weights not in first_.variables:
                        raise ValueError("cannot weight by `%s`; no such column" %weights)
                    t = []
                    for key in self.ndindex(dims=axis):
                        key.update(extra_key)
                        d = self.loc[key].values
                        t.append(d[weights])
                    weights_ = np.array(t)
                else:
                    weights_ = weights.copy()
        
            # return a copy
            toret = first_.copy()
    
            # take the mean or the sum   
            for name in first_.variables:
            
                col_data = []
                for key in self.ndindex(dims=axis):
                    key.update(extra_key)
                    d = self.loc[key].values
                    col_data.append(d[name])
                col_data = np.array(col_data)
            
                if name not in sum_only_:
                    with np.errstate(invalid='ignore'):
                        toret[name] = (col_data*weights_).sum(axis=0) / weights_.sum(axis=0)
                else:
                    toret[name] = np.sum(col_data, axis=0)
            return toret
            
        # return a SpectraSet
        if not scalar:
            
            toret = first.copy()
            for k in axis: toret = toret.drop(k)
            
            # loop over each coordinate of remaining dimensions
            for key in toret.ndindex():
                d = first.copy().sel(**key).values
                toret.loc[key] = compute_mean(d, extra_key=key)
        else:
            first = first.values
            toret = compute_mean(first)
            
        return toret
                
                
            
            
        
        

class HaloSpectraSet(xr.Dataset):
    """
    A set of `SpectraSet` instances, stored as a ``xarray.DataSet``, 
    to store the following halo spectra:
        1) Phh : halo-halo auto spectra
        2) Phm : halo-matter cross spectra
        3) Pmm : matter-matter auto spectra
    
    The class also has the ability to output the stochasticity,
    as computed from Phh, Phm, Pmm, and the linear biases.
    """
    def __init__(self, Phh, Phm, Pmm, bias, mass_keys={}):
        """
        Parameters
        ----------
        Phh : SpectraSet
            a ``SpectraSet`` storing the halo auto spectra
        Phm : SpectraSet
            a ``SpectraSet`` storing the halo-matter cross spectra
        Pmm : SpectraSet
            a ``SpectraSet`` storing the matter auto spectra
        b1  : xarray.DataArray
            a ``xarray.DataArray`` storing the linear biases
        mass_keys : dict
            a dictionary with a single key specifying the mass column name
            for the relevant auto spectra, and the matching keys
            in the cross spectra
        """        
        # intialize with Phh, Phm, Pmm, and bias
        data = {'Phh':Phh, 'Phm':Phm, 'Pmm':Pmm, 'b1':bias}
        super(HaloSpectraSet, self).__init__(data)
                
        if len(mass_keys):
            self.auto_mass_key = mass_keys.keys()[0]
            self.cross_mass_keys = mass_keys[self.auto_mass_key]
            if len(self.cross_mass_keys) != 2:
                raise ValueError("need exactly 2 keys for cross mass bins")
        else:
            self.auto_mass_key = None
            self.cross_mass_keys = []
                
    def to_lambda(self, stoch_type):
        """
        Use the halo-halo auto spectra, halo-matter cross spectra, and
        the matter-matter auto spectra to compute the stochasticity.
        There are two valid types of stochasticity:
        
            type A: 
                Lambda_A = Phh - 2*b1*Phm + b1**2 * Pmm
            type B: 
                Lambda_B = Phh - (Phm / Pmm)**2 * Pmm
        """        
        if stoch_type.lower() not in ['a', 'b']:
            raise ValueError("valid stochasticity types are `A` or `B`")

        # loop over index
        data = np.empty(self['Phh'].shape, dtype=object)
        for ii, idx in utils.ndenumerate(self['Phh'].dims, self['Phh'].coords):
            
            # get the data
            Phh = self.loc[idx]['Phh']
            Phm = self.loc[idx]['Phm']
            Pmm = self.loc[idx]['Pmm']
            b1  = self.loc[idx]['b1']
                             
            if self.auto_mass_key is not None:
                key1 = {self.auto_mass_key:idx[self.cross_mass_keys[0]]}
                key2 = {self.auto_mass_key:idx[self.cross_mass_keys[-1]]}
                Phm1, Phm2 = Phm.loc[key1], Phm.loc[key2]
                b1_1, b1_2 = b1.loc[key1], b1.loc[key2]
            else:
                Phm1 = Phm2 = Phm
                b1_1 = b1_2 = b1
            
            if any(x.isnull() for x in [Phh, Phm1, Phm2, Pmm, b1_1, b1_2]):
                continue
                
            Phh, Pmm = Phh.values, Pmm.values
            Phm1, Phm2, b1_1, b1_2 = Phm1.values, Phm2.values, b1_1.values, b1_2.values
            
            # subtract shot noise
            Phh_noshot = Phh['power'].copy()
            if self.auto_mass_key is None:
                Phh_noshot -= tools.get_Pshot(Phh)
            Pmm_noshot = Pmm['power'].copy() - tools.get_Pshot(Pmm)
            
            # stoch type B uses b1(k)
            if stoch_type.lower() == 'b':
                b1_1 = Phm1['power'].copy() / Pmm_noshot
                b1_2 = Phm2['power'].copy() / Pmm_noshot  
            lam = Phh_noshot - b1_2*Phm1['power'] - b1_1*Phm2['power'] + b1_1*b1_2*Pmm_noshot
            
            if stoch_type.lower() == 'a':
                err = (Phh['error']**2 + (b1_1*Phm1['error'])**2 + (b1_2*Phm2['error'])**2 + (b1_1*b1_2*Pmm['error'])**2)**0.5
            elif stoch_type.lower() == 'b':           
                b11_err = b1_1 * ((Phm1['error']/Phm1['power'])**2 + (Pmm['error']/Pmm_noshot)**2)**0.5
                b12_err = b1_2 * ((Phm2['error']/Phm2['power'])**2 + (Pmm['error']/Pmm_noshot)**2)**0.5
                err = Phh['error'] + (b1_1*b1_2*Pmm_noshot)*((b11_err/b1_1)**2 + (b12_err/b1_2)**2 + (Pmm['error']/Pmm_noshot)**2)**0.5
            else:
                raise ValueError("stochasticity type not recognized")

            # make a new PkResult or PkmuResult class
            power = Phh.copy()
            power['power'] = lam
            power['error'] = err                
            data[ii] = power
        
        toret = SpectraSet(data, coords=self['Phh'].coords, dims=self['Phh'].dims)
        for dim in toret.dims:
            toret = toret.dropna(dim, 'all')
        return toret
                
            

                
                
