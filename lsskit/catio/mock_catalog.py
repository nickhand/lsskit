"""
    mock_catalog.py
    lsskit.catio

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : class to hold a mock catalog of objects
"""
from . import utils, selectionlanguage, numpy as np
from pyRSD import pygcl
import pandas as pd
import operator

class MockCatalog(object):
    """
    Base class to represent an mock catalog of objects in a simulation cube
    """
    _metadata = ['redshift', 'box_size', 'units', 'cosmo']
    
    def __init__(self, redshift, box_size, units, cosmo=None, **kwargs):
        """
        Parameters
        ----------
        redshift : float
            the redshift of the simulation box
        box_size : float
            the length of the box in `Mpc` if `units` is `absolute`, or in 
            `Mpc/h` if `units` is `relative`
        units : str, {`relative`, `absolute`}
            the string specifying what type of units we have
        cosmo : {pygcl.Cosmology, str}, optional
            The cosmology of the box
        """
        if units not in ['relative', 'absolute']:
            raise ValueError("`units` should be either 'relative' or 'absolute'")
        
        self.box_size = box_size # assuming a cube here
        self.redshift = redshift 
        self.units    = units # distance units, either "relative" or "absolute"
                
        # the cosmology
        self.cosmo = cosmo
        if self.cosmo is not None:
            if not isinstance(self.cosmo, (basestring, pygcl.Cosmology)): 
                raise TypeError("`Cosmo` must be one of [str, pygcl.Cosmology]")
            if isinstance(self.cosmo, basestring): 
                self.cosmo = pygcl.Cosmology(self.cosmo)
            
        # this will be a pandas data frame holding all the info
        self._data = None
        
        # this is a DataFrame holding a subsample of `self._data`
        self._sample = None
        
        # this holds any restrictions we've used
        self.restrictions = Sample()
        
        # subsample indices
        self.subsample_indices = None
        
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
            self._metadata.append(k)

    @classmethod
    def from_dataframe(cls, data, **meta):
        """
        Return a MockCatalog instance from a `pandas.DataFrame` and
        a dictionary of metadata
        
        Notes
        -----
        There are 3 required metadata keywords: `redshift`, `box_size`,
        and `units`
        
        Parameters
        ----------
        data : pandas.DataFrame
            the mock catalog data, stored as a dataframe
        meta : key, value pairs
            keyword arguments specifying the metadata for the class
        """
        required = ['redshift', 'box_size', 'units']
        for k in required:
            if k not in meta:
                raise ValueError("`%s` must be provided as a keyword" %k)
        mock = cls(meta.pop('redshift'), meta.pop('box_size'), meta.pop('units'), **meta)
        mock._data = data
        return mock
    
    @classmethod
    def from_hdf(cls, filename):
        """
        Return a MockCatalog instance as read from a HDF5 file
        
        Parameters
        ----------
        filename : str
            the name of the HDF5 file
        """
        with pd.HDFStore(filename) as store:
            df = store['data']
            meta = store.get_storer('data').attrs.metadata
        
        return cls.from_dataframe(df, **meta)
        
    def to_hdf(filename):
        """
        Write the class out to a HDF5 file
        
        Parameters
        ----------
        filename : str
            the name of the file to save the MockCatalog instance to
        """
        meta = {k:getattr(self,k) for k in self._metadata if hasattr(self, k)}
        store = pd.HDFStore(filename)
        store.put('data', self._data)
        store.get_storer('data').attrs.metadata = meta
        store.close()
        
    #---------------------------------------------------------------------------
    # read-only attributes
    #---------------------------------------------------------------------------
    @property
    def mass_restricted(self):
        """
        Return `True` if the sample has been restricted via `restrict_by_mass_pdf`
        """
        try:
            return self._mass_restricted
        except AttributeError:
            return False
            
    @property
    def index_restricted(self):
        """
        Return `True` if the sample has been restricted via `restrict_by_index`
        """
        try:
            return self._index_restricted
        except AttributeError:
            return False
            
    @property
    def restricted(self):
        """
        Return `True` if the sample is restricted
        """
        return not self.restrictions.is_clear() or self.mass_restricted or self.index_restricted
        
    @property
    def sample(self):
        """
        DataFrame holding the object info for the current sample
        """
        if not self.restricted:
            if self.subsample_indices is None:
                return self._data
            else:
                return self._data.iloc[self.subsample_indices]
        else:
            if self.subsample_indices is None:
                return self._sample
            else:
                return self._sample.iloc[self.subsample_indices]
            
    @property
    def _sample_total(self):
        """
        Total number of objects in the current sample
        """
        return len(self.sample)

    #---------------------------------------------------------------------------
    # utility functions
    #---------------------------------------------------------------------------
    def random_subsample(self, N):
        """
        Set the indices defining a random subsample of size `N` from the 
        current object sample
        """
        if N > self._sample_total:
            msg = "Cannot select subsample of size %d from galaxy sample of size %d" %(N, self._sample_total)
            raise ValueError(msg)
        self.subsample_indices = sorted(np.random.choice(xrange(self._sample_total), N, replace=False))
    
    def clear_subsample(self):
        """
        Clear the random subsample selection so we are using the full sample
        """
        self.subsample_indices = None

    def clear_restrictions(self):
        """
        Clear any restrictions
        """
        self.restrictions.clear_flags()
        self._mass_restricted = self._index_restricted = False
        self._sample = None
    
    def restrict_by_index(self, index):
        """
        Restrict the sample size by the `objid` index
        """
        if not isinstance(index, pd.Index):
            raise TypeError("To restrict by index, please provide a pandas Index")
            
        self._sample = self._data.loc[index]
        self._index_restricted = True
                    
    def restrict_by_mass_pdf(self, mass_pdf, mass_col='mass', unique_col=None,
                                bins=None, total=None):
        """
        Restrict the sample size by the `objid` index
        """
        self._mass_restricted = False
        
        # first, get the masses
        masses = self.sample[mass_col]
        
        # remove any non-unique masses, if so desired
        if unique_col is not None:
            groups = self.sample.groupby(unique_col)
            masses = groups[mass_col].first()
            
        # get the objids of the chosen ones
        index = utils.sample_by_mass_pdf(masses, mass_pdf, bins=bins, N=total)
        
        # restrict the sample
        self._sample = self._data.loc[index]
        self._mass_restricted = True
                    
    @classmethod
    def from_ascii(cls, filename, info_dict, object_types=None, skip_lines=0, **meta):
        """
        Create a MockCatalog instance by reading object information from an 
        ASCII file
        
        Notes
        -----
        There are 3 required metadata keywords: `redshift`, `box_size`,
        and `units`
        
        Parameters
        ----------
        filename : str
            The name of the file containing the info to load into each halo
        info_dict : dict
            A dictionary with keys corresponding to the names of the columns
            in the `DataFrame` and the values corresponding to the column 
            numbers to read from the input file
        object_types : dict, optional
            A dictionary holding parameters to distinguish between 
            objects of different types. The `column` key should give
            the name of the column holding the type information and
            the `types` key holds a dict of the different types            
        skip_lines : int, optional
            The number of lines to skip when reading the input file; 
            default is 0
        meta : key, value pairs, optional
            keyword arguments specifying the metadata for the class
        """        
        # sort the fields we are reading by the column numbers (the values of info_dict)
        sorted_info = sorted(info_dict.items(), key=operator.itemgetter(1))
        col_names, col_nums = map(list, zip(*sorted_info))

        # use pandas to efficiently read the data into a data frame
        kwargs = {}
        kwargs['engine']           = 'c'
        kwargs['skiprows']         = skip_lines
        kwargs['header']           = None
        kwargs['delim_whitespace'] = True
        kwargs['usecols']          = col_nums
        kwargs['names']            = col_names 
        
        print "reading ascii file..."
        data = pd.read_csv(filename, **kwargs)
        data.index.name = 'objid'
        print "  ...done"
                    
        # store the object type info, if provided
        if object_types is not None:                
            type_col = object_types.get('column', None)
            if type_col is not None:
                
                # replace with the appropriate types
                type_values = data[type_col]
                types = object_types['types']
                new_types = data[type_col].replace(to_replace=types.values(), value=types.keys())
                
                # delete the old column and add the new one
                del data[type_col]
                data['type'] = new_types
                
        return cls.from_dataframe(data, **meta)
                        
    def to_coordinates(self, filename, fields, units=None, header=[], 
                            temporary=False, replace_with_nearest=False):
        """
        Write the coordinates of the mock catalog out to a file
        
        Parameters
        ----------
        filename : str
            The name of the file to save the coordinates to
        fields : list
            A list of the names of the attributes to write to the file
        units : str, {`absolute`, `relative`}, optional
            The name of the units of the coordinates to output
        header : list, optional
            List of strings that will be written as a header, one per line. Any 
            strings beginning with `$` will have the `MockHOD` attribute 
            replace the string
        temporary : bool, optional
            If `True`, write the results out to a `tempfile.NamedTemporaryFile`
        replace_with_nearest : {str, bool}, optional
            Replace the values of all collided galaxies with those of the 
            nearest neighbor on the sky -- this is correcting for 
            fiber collisions by double counting nearest neighbors
            
        Returns
        -------
        filename : str
            The name of the file the coordinates were written to
        """         
        if header is None: header = []   
       
        # create the output file
        if temporary:
            import tempfile
            outfile = tempfile.NamedTemporaryFile(delete=False) 
            out_name = outfile.name
        else:
            outfile = open(filename, 'w')
            out_name = filename
        
        # get the units conversion factor
        conversion_factor = 1.
        if units is not None:
            conversion_factor = utils.h_conversion_factor('distance', self.units, units, self.cosmo['h'])
            
        # write out the header
        header_copy = list(header)
        if len(header_copy) > 0:
            
            for i, header_line in enumerate(header_copy):
                line = header_line['line']
                fmt = header_line.get('fmt', '%s')
                if '$' in line:
                    words = [word for word in line.split() if word.startswith('$')]
                    for word in words:
                        varname = word.split('$')[-1]
                        var = getattr(self, varname)
                        if varname == 'box_size': var *= conversion_factor
                        
                        # now replace
                        header_copy[i]['line'] = line.replace(word, fmt %var)
                    
            hdr = "%s\n" %('\n'.join(h['line'] for h in header_copy))
            outfile.write(hdr)
            
        # close the file  
        outfile.close()
          
        # get the output fields, optionally replacing with nearest neighbor values
        if replace_with_nearest is None or not replace_with_nearest:
            output_fields = self.sample[fields]*conversion_factor
        else:
            output_fields = self._replace_with_nearest(replace_with_nearest, fields)*conversion_factor
        
        # now write out
        output_fields.to_csv(out_name, sep=" ", header=False, index=False, mode='a')
        return out_name    
  
        
#-------------------------------------------------------------------------------
class Sample(object):
    """
    A class to handle sample restrictions
    """
    def __init__(self):
        self.flags = {}
    
    def __str__(self):
        
        keys = [k + " flags" for k in self.flags.keys()]
        toret = "\n".join("%-15s %s" %(k, v) for k, v in zip(keys, self.flags.values()))
        return toret
        
    def is_clear(self):
        """
        Check if the sample has no restrictions
        """
        return len(self.flags) == 0
    
    def clear_flags(self, key=None):
        """
        Clear any restrictions. If `key` is `None`, clear all restrictions, 
        otherwise, only remove the restrictions of type `key`
        """
        if key is None: 
            self.flags.clear()
        else:
            if key not in self.flags:
                raise ValueError("Cannot clear restrictions of type `%s`; none present" %key)
            self.flags.pop(key)
            
    def set_flags(self, key, condition):
        """
        Set the boolean conditions of type `key`
        """
        self.flags[key] = selectionlanguage.Query(condition)
    
    def slice_frame(self, frame):
        """
        Return a sliced DataFrame corresponding to the boolean conditions
        set by all flags in `self.flags`
        """
        # initialize condition to unity for all rows
        condition = np.ones(len(frame), dtype=bool)
        
        # get the condition for each flag in self.flags
        for flag in self.flags.values():
            condition = np.logical_and(condition, flag.get_mask(frame))
        
        return frame[condition]
    
        
