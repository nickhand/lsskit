import xray 
import itertools

from . import utils, readers
from .. import numpy as np

class SpectraSet(xray.DataArray):
    """
    N-dimensional set of power spectra measurements
    """

    def __init__(self, data, coords=None, dims=None, name=None,
                 attrs=None):
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
        super(SpectraSet, self).__init__(data, coords=coords, dims=dims, name=name, attrs=attrs)
        
 
    @classmethod
    def from_files(cls, result_dir, basename, coords, dims=None, **kwargs):
        """
        Return a SpectraSet instance by loading data from all files in 
        ``result_dir`` with base ``basename``. The filename is formatting using 
        the specified coordinates and dimensions
        
        Parameters
        ----------
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
        """
        if dims is None:
            if not utils.is_dict_like(coords):
                raise TypeError("if no `dims` provided, `coords` must be a dict")
            dims = coords.keys()
            coords = coords.values()
        if len(dims) != len(coords):
            raise ValueError("shape mismatch between supplied `dims` and `coords`")
            
        data = np.empty(map(len, coords), dtype=object)
        for i, f in utils.enum_files(result_dir, basename, dims, coords):
            data[i] = readers.load_data(f)
        return SpectraSet(data, coords=coords, dims=dims, **kwargs)


    def ndindex(self, dims=None):
        """
        A generator to iterate over the specified dimensions, yielding a
        dictionary holding the index keys
        
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
            key = {k:v.values.tolist() for k,v in self.coords.iteritems()}
            yield key
        else:
            for d in utils.ndindex(dims, self.coords):
                val = self.loc[d]
                key = {k:v.values.tolist() for k,v in val.coords.iteritems()}
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
            key = {k:v.values.tolist() for k,v in self.coords.iteritems()}
            yield key, self
        else:
            for d in utils.ndindex(dims, self.coords):
                val = self.loc[d]
                key = {k:v.values.tolist() for k,v in val.coords.iteritems()}
                yield key, val
            
    def add_errors(self, power_x1=None, power_x2=None):
        """
        Add power spectrum errors to each object in the set
        
        Parameters
        ----------
        power_x1, power_x2 : nbodykit.PkmuResult, nbodykit.PkResult, optional
            If the set stores a cross-power measurement, the auto power
            measurements are needed to compute the error
        """
        p1 = p2 = None
        for coord, power in self.nditer():
            power = power.values
            if power_x1 is not None:
                coord1 = {k:coord[k] for k in power_x1.dims}
                p1 = power_x1.loc[coord1].values
            if power_x2 is not None:
                coord2 = {k:coord[k] for k in power_x2.dims}
                p2 = power_x2.loc[coord2].values
            utils.add_errors(power, p1, p2)


class HaloSpectraSet(xray.Dataset):
    """
    A set of `SpectraSet` instances, stored as a ``xray.DataSet``, 
    to store the following halo spectra:
        1) Phh : halo-halo auto spectra
        2) Phm : halo-matter cross spectra
        3) Pmm : matter-matter auto spectra
    
    The class also has the ability to output the stochasticity,
    as computed from Phh, Phm, Pmm, and the linear biases.
    """
    def __init__(self, Phh, Phm, Pmm, bias):
        """
        Parameters
        ----------
        Phh : SpectraSet
            a ``SpectraSet`` storing the halo auto spectra
        Phm : SpectraSet
            a ``SpectraSet`` storing the halo-matter cross spectra
        Pmm : SpectraSet
            a ``SpectraSet`` storing the matter auto spectra
        b1  : xray.DataArray
            a ``xray.DataArray`` storing the linear biases
        """
        # first add errors
        Phh.add_errors()
        Pmm.add_errors()
        Phm.add_errors(Phh, Pmm)
        
        # intialize with Phh, Phm, Pmm, and bias
        data = {'Phh':Phh, 'Phm':Phm, 'Pmm':Pmm, 'b1':bias}
        super(HaloSpectraSet, self).__init__(data)
        
    @classmethod
    def load_masses(cls, filename, dims, shape):
        """
        Load a pickled dictionary of average halo masses and return
        a `xray.DataArray`
        
        Parameters
        ----------
        filename : str
            the name of the file holding the pickled data
        dims : list of str
            the list of strings corresponding to the names of
            the dimensions of the dictionary keys
        shape : list of int
            the shape of the data values, corresponding to the
            shape of dim 0, dim 1, etc
        """
        import pickle
        
        # load the data
        masses = pickle.load(open(filename))
            
        # sort keys and values by the keys
        keys = masses.keys()
        M = masses.values()
        sorted_lists = sorted(zip(keys, M), key=lambda x: x[0])
        keys, M = [[x[i] for x in sorted_lists] for i in range(2)]

        # make the coords and return a DataArray
        coords = zip(*keys)
        coords = [np.unique(x) for x in coords]
        return xray.DataArray(np.array(M).reshape(shape), coords, dims)
        
    @classmethod
    def load_biases(cls, filename, dims, shape):
        """
        Load a pickled dictionary of linear biases and return
        a `xray.DataArray`
        
        Parameters
        ----------
        filename : str
            the name of the file holding the pickled data
        dims : list of str
            the list of strings corresponding to the names of
            the dimensions of the dictionary keys
        shape : list of int
            the shape of the data values, corresponding to the
            shape of dim 0, dim 1, etc
        """
        import pickle
        
        # load the data
        biases = pickle.load(open(filename))
            
        # sort keys and values by the keys
        keys = biases.keys()
        b1 = biases.values()
        sorted_lists = sorted(zip(keys, b1), key=lambda x: x[0])
        keys, b1 = [[x[i] for x in sorted_lists] for i in range(2)]

        # make the coords and return a DataArray
        coords = zip(*keys)
        coords = [np.unique(x) for x in coords]
        return xray.DataArray(np.array(b1).reshape(shape), coords, dims)
        
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
        from nbodykit import pkmuresult, pkresult
        
        if stoch_type.lower() not in ['a', 'b']:
            raise ValueError("valid stochasticity types are `A` or `B`")

        # loop over index
        data = np.empty(self['Phh'].shape, dtype=object)
        for ii, idx in utils.ndenumerate(self.dims, self.coords):
            
            # get the data
            Phh = self.loc[idx]['Phh'].values
            Phm = self.loc[idx]['Phm'].values
            Pmm = self.loc[idx]['Pmm'].values
            b1  = self.loc[idx]['b1'].values
            
            # subtract shot noise
            Phh_noshot = Phh['power'] - Phh.box_size**3/Phh.N1
            Pmm_noshot = Pmm['power'] - Pmm.box_size**3/Pmm.N1
            
            # stoch type A
            if stoch_type.lower() == 'a':
                lam = Phh_noshot - 2*b1*Phm['power'] + b1**2*Pmm_noshot
                err = (Phh['error']**2 + (2*b1*Phm['error'])**2 + (b1**2*Pmm['error'])**2)**0.5
            # stoch type B
            elif stoch_type.lower() == 'b':           
                b1k = Phm['power'] / Pmm_noshot
                b1k_err = b1k * ((Phm['error']/Phm['power'])**2 + (Pmm['error']/Pmm_noshot)**2)**0.5
                lam = Phh_noshot - b1k**2 * Pmm_noshot
                err = (Phh['error']**2 + (2*b1k*Pmm_noshot*b1k_err)**2 + (b1k*Pmm['error'])**2)**0.5
            else:
                raise ValueError("stochasticity type not recognized")

            # make a new PkResult or PkmuResult class
            d = {}
            d['power'] = lam
            d['error'] = err
            if isinstance(Phh, pkresult.PkResult):
                d['k'] = Phh['k']
                power = pkresult.PkResult(Phh.kedges, d)
            elif isinstance(Phh, pkmuresult.PkmuResult):
                d['k'] = Phh['k']
                d['mu'] = Phh['mu']
                power = pkresult.PkResult(Phh.kedges, Phh.muedges, d)
                
            data[ii] = power
        
        return SpectraSet(data, coords=self.coords, dims=self.dims)
                
            

                
                
