"""
 mock_catalog.py
 lss: class to hold a mock catalog of objects
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/26/2015
"""
from . import numpy as np, tools
import copy
import tempfile
import operator
import pandas as pd
import pyparsing as pp
from pyRSD import pygcl

#-------------------------------------------------------------------------------

# Note: operator here gives the function needed to go from 
# `absolute` to `relative` units
variables = {"wavenumber" : {'operator': operator.div, 'power' : 1}, \
             "distance" : {'operator': operator.mul, 'power' : 1}, \
             "volume" : {'operator': operator.mul, 'power' : 3}, \
             "power" : {'operator': operator.mul, 'power' : 3} }

#-------------------------------------------------------------------------------
def h_conversion_factor(variable_type, input_units, output_units, h):
    """
    Return the factor needed to convert between the units, dealing with 
    the pesky dimensionless Hubble factor, `h`.
    
    Parameters
    ----------
    variable_type : str
        The name of the variable type, must be one of the keys defined
        in ``units.variables``.
    input_units : str, {`absolute`, `relative`}
        The type of the units for the input variable
    output_units : str, {`absolute`, `relative`}
        The type of the units for the output variable
    h : float
        The dimensionless Hubble factor to use to convert
    """
    units_types = ['relative', 'absolute']
    units_list = [input_units, output_units]
    if not all(t in units_types for t in units_list):
        raise ValueError("`input_units` and `output_units` must be one of %s, not %s" %(units_types, units_list))
        
    if variable_type not in variables.keys():
        raise ValueError("`variable_type` must be one of %s" %variables.keys())
    
    if input_units == output_units:
        return 1.
    
    exponent = variables[variable_type]['power']
    if input_units == "absolute":
        return variables[variable_type]['operator'](1., h)**(exponent)
        
    else:
        return 1./(variables[variable_type]['operator'](1., h)**(exponent))

#-------------------------------------------------------------------------------
def load_catalog(filename):
    """
    Load a ``MockCatalog`` object from a HDF5 file using the pandas module
    
    Parameters
    ----------
    filename : str
        The name of the pickled file
    
    Returns
    -------
    mock : MockHOD object
        The MockHOD class
    """
    store = pd.HDFStore(filename, 'r')
    mock = store.get_storer('data').attrs.catalog
    mock._data = store['data']
    store.close()
    
    # update the sample
    if not mock.restrictions.is_clear():
        mock._sample = mock.restrictions.slice_frame(mock._data)
    return mock

#-------------------------------------------------------------------------------
class MockCatalog(object):
    """
    Base class to represent an mock catalog of objects in a simulation cube
    """
    
    def __init__(self, redshift, box_size, units, cosmo=None):
        """
        Parameters
        ----------
        redshift : float
            The redshift of the simulation box
        box_size : float
            The length of the box in `Mpc` if `units` is `absolute`, or in 
            `Mpc/h` if `units` is `relative`
        units : str, {`relative`, `absolute`}
            The string specifying what type of units we have
        cosmo : {pygcl.Cosmology, str}, optional
            The cosmology of the box
        """
        if units not in ['relative', 'absolute']:
            raise ValueError("Input units not understood; must be one of ['relative', 'absolute']")
        
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

    #---------------------------------------------------------------------------
    @property
    def sample(self):
        """
        DataFrame holding the object info for the current sample
        """
        if self.restrictions.is_clear():
            if self.subsample_indices is None:
                return self._data
            else:
                return self._data.iloc[self.subsample_indices]
        else:
            if self.subsample_indices is None:
                return self._sample
            else:
                return self._sample.iloc[self.subsample_indices]
            
    #---------------------------------------------------------------------------
    @property
    def _sample_total(self):
        """
        Total number of objects in the current sample
        """
        return len(self.sample)

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
    
    #---------------------------------------------------------------------------
    def clear_subsample(self):
        """
        Clear the random subsample selection so we are using the full sample
        """
        self.subsample_indices = None

    #---------------------------------------------------------------------------
    def clear_restrictions(self):
        """
        Clear any restrictions
        """
        self.restrictions.clear_flags()
        self._sample = None
    
    #---------------------------------------------------------------------------
    def restrict_by_index(self, index):
        """
        Restrict the sample size by the `objid` index
        """
        if not isinstance(index, pd.Index):
            raise TypeError("To restrict by index, please provide a pandas Index")
            
        self._sample = self._data.loc[index]
                    
    #---------------------------------------------------------------------------
    def restrict_by_mass_pdf(self, mass_pdf, mass_col='mass', unique_col=None,
                                bins=None, total=None):
        """
        Restrict the sample size by the `objid` index
        """
        # first, get the masses
        masses = self.sample[mass_col]
        
        # remove any non-unique masses, if so desired
        if unique_col is not None:
            groups = self.sample.groupby(unique_col)
            masses = groups[mass_col].first()
            
        # get the objids of the chosen ones
        index = tools.sample_by_mass_pdf(masses, mass_pdf, bins=bins, N=total)
            
        # restrict the sample
        self._sample = self._data.loc[index]
                    
    #---------------------------------------------------------------------------
    def load(self, filename, info_dict, **kwargs):
        """
        Load objects from a file
        
        Parameters
        ----------
        filename : str
            The name of the file containing the info to load into each halo
        info_dict : dict
            A dictionary with keys corresponding to the names of the columns
            in the `DataFrame` and the values corresponding to the column 
            numbers to read from the input file
        kwargs : dict
            Optional keyword arguments include: 
                extra_info : dict
                    Dictionary of values to store as attributes of `self`
                object_types : dict
                    A dictionary holding parameters to distinguish between 
                    objects of different types. The `column` key should give
                    the name of the column holding the type information and
                    the `types` key holds a dict of the different types            
                skip_lines : int
                    The number of lines to skip when reading the input file; 
                    default is 0
        """        
        # get the default keywords
        extra_info   = kwargs.get('extra_info', None)
        object_types = kwargs.get('object_types', None)
        skip_lines   = kwargs.get('skip_lines', 0)

        # save any extra info
        if extra_info is not None:
            for k, v in extra_info.iteritems():
                setattr(self, k, v)
        
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
        
        print "reading mock catalog..."
        self._data = pd.read_csv(filename, **kwargs)
        self._data.index.name = 'objid'
        print "...done"
                    
        # store the object type info, if provided
        if object_types is not None:                
            type_col = object_types.get('column', None)
            if type_col is not None:
                
                # replace with the appropriate types
                type_values = self._data[type_col]
                types = object_types['types']
                new_types = self._data[type_col].replace(to_replace=types.values(), value=types.keys())
                
                # delete the old column and add the new one
                del self._data[type_col]
                self._data['type'] = new_types
                        
    #---------------------------------------------------------------------------
    def write_coordinates(self, filename, fields, units=None, header=[], 
                            temporary=False, replace_with_nearest=False):
        """
        Write the coordinates of the specified galaxy sample out to a file
        
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
            outfile = tempfile.NamedTemporaryFile(delete=False) 
            out_name = outfile.name
        else:
            outfile = open(filename, 'w')
            out_name = filename
        
        # get the units conversion factor
        conversion_factor = 1.
        if units is not None:
            conversion_factor = h_conversion_factor('distance', self.units, units, self.cosmo['h'])
            
        # write out the header
        header_copy = copy.deepcopy(header)
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
            
        # get the output fields, optionally replacing with nearest neighbor values
        if replace_with_nearest is None or not replace_with_nearest:
            output_fields = self.sample[fields]*conversion_factor
        else:
            output_fields = self._replace_with_nearest(replace_with_nearest, fields)*conversion_factor
        
        # now write out the np array
        np.savetxt(outfile, output_fields)
        outfile.close()        
        return out_name    
  
    #---------------------------------------------------------------------------
    def save(self, filename):
        """
        Save the `MockCatalog` instance as a HDF5 file to the filename specified
        
        Parameters
        ----------
        filename : str 
            the filename to output to
        """
        print "saving mock catalog..."
        store = pd.HDFStore(filename, 'w')
        store['data'] = self._data
        self._data = None
        self._sample = None
        store.get_storer('data').attrs.catalog = self
        store.close()
    
    #---------------------------------------------------------------------------
    def _compute_pdf(self, x, log=False, N_bins=50):
        """
        Internal function to compute the probability distribution function,
        `dn / dx` (or `dn/dlog10x`) of the input array
        
        Parameters
        ----------
        x : numpy.ndarray
            the array of values to compute dn/dx from
        log : bool, optional
            if `True`, return the dn/dlnx instead
        """
        from matplotlib import pyplot as plt
        
        min_x = np.amin(x)
        max_x = np.amax(x)

        # first plot and then we normalize
        x_bins = np.logspace(np.log10(min_x), np.log10(max_x), N_bins)
        pdf, bins, patches = plt.hist(x, bins=x_bins)
        bincenters = 0.5*(bins[1:] + bins[:-1])
        plt.cla()
        plt.close()
        
        # transform N(M) into dN/dlnM
        widths = np.diff(bins)
        pdf = 1.*pdf/sum(pdf)
        if log:
            dlogx = np.diff(np.log10(bins))
            pdf /= dlogx
        else:
            dx = np.diff(bins)
            pdf /= dx

        return bins, pdf, widths
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class SampleFlags(object):
    
    comparison_operators = {'<' : operator.lt, '<=' : operator.le, 
                            '>' : operator.gt, '>=' : operator.ge, 
                            '==' : operator.eq, '!=' : operator.ne, 
                            'and' : operator.and_, 'or' : operator.or_,
                            'not' : operator.not_ }
    
    def __init__(self, str_condition):
                
        # save the string condition and parse it
        self.string_condition = str_condition
        self._parse_condition(self.string_condition)
    
    #---------------------------------------------------------------------------
    def __iter__(self):
        return iter(self.condition)
            
    #---------------------------------------------------------------------------
    def _parse_condition(self, str_condition):
        """
        Parse the input string condition
        """        
        # set up the regex for the individual terms
        operator = pp.Regex(">=|<=|!=|>|<|==").setName("operator")
        number = pp.Regex(r"[+-]?\d+(:?\.\d*)?(:?[eE][+-]?\d+)?")
        identifier = pp.Word(pp.alphanums, pp.alphanums + "_")

        # look for things like key (operator) value
        condition = pp.Group(identifier + operator + (number|identifier) )
        expr = pp.operatorPrecedence(condition, [ ("not", 1, pp.opAssoc.RIGHT, ), 
                                                  ("and", 2, pp.opAssoc.LEFT, ), 
                                                  ("or", 2, pp.opAssoc.LEFT, ) ])

        self.condition = expr.parseString(str_condition)[0]
              
    #---------------------------------------------------------------------------
    def __str__(self):
        return self.string_condition
        
    #---------------------------------------------------------------------------

#-------------------------------------------------------------------------------
class Sample(object):
    """
    A class to handle sample restrictions
    """
    def __init__(self):
        self.flags = {}
    
    #---------------------------------------------------------------------------
    def __str__(self):
        
        keys = [k + " flags" for k in self.flags.keys()]
        toret = "\n".join("%-15s %s" %(k, v) for k, v in zip(keys, self.flags.values()))
        return toret
        
    #---------------------------------------------------------------------------
    def is_clear(self):
        """
        Check if the sample has no restrictions
        """
        return len(self.flags) == 0
    
    #---------------------------------------------------------------------------
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
            
    #---------------------------------------------------------------------------
    def set_flags(self, key, condition):
        """
        Set the boolean conditions of type `key`
        """
        self.flags[key] = SampleFlags(condition)
    
    #---------------------------------------------------------------------------    
    def slice_frame(self, frame):
        """
        Return a sliced DataFrame corresponding to the boolean conditions
        set by all flags in `self.flags`
        """
        # initialize condition to unity for all rows
        condition = np.ones(len(frame), dtype='bool')
        
        # get the condition for each flag in self.flags
        for flag in self.flags.values():
            condition = np.logical_and(condition, self._valid(frame, flag.condition))
        
        return frame[condition]
    
    #---------------------------------------------------------------------------
    def _valid(self, frame, flags):
        """
        Return `True` if this halo is within the sample, `False` otherwise
        """
        # return the string representation
        if isinstance(flags, str):
            return flags
        # the flags is a pyparsing.ParseResults
        else:
            # this is a not clause
            if len(flags) == 2:
                
                # only need operator and value
                operator = SampleFlags.comparison_operators[flags[0]]
                value = self._valid(frame, flags[-1])
                ans = operator(value)
            # this is a and/or clause
            else:
                
                if not isinstance(flags, pp.ParseResults):
                    raise TypeError("Error reading pyparsing results")
                
                # get the key, value and operator function
                key = self._valid(frame, flags[0])
                value = self._valid(frame, flags[-1])
                operator = SampleFlags.comparison_operators[flags[1]]

                # now get the frame attributes
                if isinstance(key, str): 
                    key = getattr(frame, key)
                if isinstance(value, str): 
                    try:
                        value = float(value)
                    except:
                        pass    
                ans = operator(key, value)
                
            return ans
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------
        
