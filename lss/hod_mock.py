"""
 hod_mock.py
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 06/13/2014
"""
import numpy as np
import copy
import tempfile
import operator
import pandas as pd
from scipy.spatial.distance import pdist, squareform
try:
    from scipy.spatial import cKDTree as KDTree
except:
    from scipy.spatial import KDTree
import pyparsing as pp

from utils import utilities
import plotify as pfy
from . import angularFOF
from cosmology.parameters import Cosmology, default_params
from cosmology import evolution
from cosmology.utils.units import h_conversion_factor

#-------------------------------------------------------------------------------
def load(filename):
    """
    Load a ``MockHOD`` object from a HDF5 file using the pandas module
    
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
    
    mock = store.get_storer('galaxies').attrs.my_attribute
    mock._galaxies = store['galaxies']
    store.close()
    
    # update the sample
    if not mock.restrictions.is_clear():
        mock._sample = mock.restrictions.slice_frame(mock._galaxies)
    return mock
#end load

#-------------------------------------------------------------------------------
class MockHOD(object):
    """
    Class to represent an HOD mock catalog in a simulation cube
    """
    
    def __init__(self, redshift, box_size, units, cosmo=None):
        
        if units not in ['relative', 'absolute']:
            raise ValueError("Input units not understood; must be one of ['relative', 'absolute']")
        
        self.box_size = box_size # assuming a cube here
        self.redshift = redshift 
        self.units    = units # distance units, either "relative" or 
        
        # keep track of total number of galaxies
        self._total_galaxies = 0
        
        # store the cosmology as a Cosmology class
        if cosmo is None:
            cosmo = default_params
        self.cosmo = Cosmology(cosmo)
        
        # this will be a pandas data frame holding all the info
        self._galaxies = None
        
        # this is a DataFrame holding a subsample of `self.galaxies`
        self._sample = None
        
        # this holds any restrictions we've used
        self.restrictions = Sample()
        
        # subsample indices
        self.subsample_indices = None
    #end __init__

    #---------------------------------------------------------------------------
    # READ-ONLY PROPERTIES
    #---------------------------------------------------------------------------
    @property
    def sample(self):
        """
        DataFrame holding the galaxy info for the current sample
        """
        if self.restrictions.is_clear():
            if self.subsample_indices is None:
                return self._galaxies
            else:
                return self._galaxies.iloc[self.subsample_indices]
        else:
            if self.subsample_indices is None:
                return self._sample
            else:
                return self._sample.iloc[self.subsample_indices]
            
    #---------------------------------------------------------------------------
    @property
    def total_galaxies(self):
        """
        Total number of galaxies in the mock catalog
        """
        return len(self.sample)

    #---------------------------------------------------------------------------
    @property
    def potentially_collided(self):
        """
        Total number of potentially collided galaxies in the current sample
        """
        return len(self.sample[self.sample.collided == 1])
    
    #---------------------------------------------------------------------------
    @property
    def total_collided(self):
        """
        Total number of collided galaxies in the current sample
        """
        cond = (self.sample.collided == 1)&(self.sample.resolved == 0)
        return len(self.sample[cond])
    
    #---------------------------------------------------------------------------
    @property
    def total_resolved(self):
        """
        Total number of total resolved, collided galaxies in the current sample
        """
        return len(self.sample[self.sample.resolved == 1])
    
    #---------------------------------------------------------------------------
    @property
    def total_halos(self):
        """
        Total number of halos in the mock catalog
        """
        return len(self.sample.groupby(self.halo_id).groups)

    #---------------------------------------------------------------------------
    @property
    def cen_sat_pairs(self):
        """
        The number of cen-sat pairs for use in computing the amplitude of the
        central - satellite one-halo term, which is related by
        
        :math: P1halo = cen_sat_pairs / V / nbar**2
        
        The expression used here is:
        
        :math: 2 * \sum N_{sat, i}
        
        where the sum is over all halos with a central galaxy and at least one
        satellite galaxy
        """
        halos = self.sample.groupby(self.halo_id)
        if not hasattr(halos, 'N_cen') or not hasattr(halos, 'N_sat'):
            return 0.
        N_cen = halos.N_cen.first()
        N_sat = halos.N_sat.first()
        
        return 2*N_sat[(N_cen > 0)&(N_sat > 0)].sum()
        
    #---------------------------------------------------------------------------
    @property
    def sat_sat_pairs(self):
        """
        The number of sat-sat pairs for use in computing the amplitude of the
        satellite - satellite one-halo term, which is related by
        
        :math: P1halo = sat_sat_pairs / V / nbar**2
        
        The expression used here is:
        
        :math: 2 * \sum N_{sat, i} * (N_{sat, i} - 1)
        
        where the sum is over all halos with a central galaxy and more than one
        satellite galaxy
        
        """
        halos = self.sample.groupby(self.halo_id)
        if not hasattr(halos, 'N_sat'):
            return 0.
        N_sat = halos.N_sat.first()
        
        if not hasattr(halos, 'N_cen'):
            inds = N_sat > 1
        else:
            N_cen = halos.N_cen.first()
            inds = (N_sat > 1)&(N_cen > 0)
        return (N_sat[inds]*(N_sat[inds]-1)).sum()
        
    #---------------------------------------------------------------------------
    @property
    def nearest_neighbor_ids(self):
        """
        Return an Index with the galaxy ids of the nearest neighbors of all
        collided and unresolved galaxies
        """
        try:
            return self._nearest_neighbor_ids
        except:
            self._find_nearest_neighbors()
            return self._nearest_neighbor_ids
    
    #---------------------------------------------------------------------------
    @property
    def collided_unresolved_ids(self):
        """
        Return an Index with the galaxy ids of all collided and 
        unresolved galaxies
        """
        try:
            return self._collided_unresolved_ids
        except:
            self._find_nearest_neighbors()
            return self._collided_unresolved_ids    
    #---------------------------------------------------------------------------
    # Function calls    
    #---------------------------------------------------------------------------
    def _find_nearest_neighbors(self):
        """
        Compute the nearest neighbors of all collided/unresolved galaxies
        """
        # we can double count using any uncollided galaxies (for which
        # we have a redshift)
        cond = (self._galaxies.collided == 0)|(self._galaxies.resolved==1)
        uncollided_gals = self._galaxies[cond]
    
        # initialize the kdtree for NN calculations
        tree = KDTree(uncollided_gals[self.coord_keys])
    
        # find the NN for only the collided galaxies
        cond = (self.sample.collided == 1)&(self.sample.resolved==0)
        collided_gals = self.sample[cond]
        dists, inds = tree.query(collided_gals[self.coord_keys], k=1)
    
        self._collided_unresolved_ids = collided_gals.index
        self._nearest_neighbor_ids    = uncollided_gals.iloc[inds].index
    #end _find_nearest_neighbors
    
    #---------------------------------------------------------------------------
    def info(self):
        """
        Print out the sample info
        """
        N_tot = self.total_galaxies
        print "total number of galaxies = %d" %N_tot
        print "total number of halos = %d" %self.total_halos
        print 
        
        N_c, N_cA, N_cB = self.centrals_totals()
        if N_c > 0:
            print "centrals"
            print "----------------------"
            print "overall fraction = %.3f" %(1.*N_c/N_tot)
            print "fraction without satellites in same halo = %.3f" %(1.*N_cA/N_c)
            print "fraction with satellites in same halo = %.3f" %(1.*N_cB/N_c)
            print 
        
        N_s, N_sA, N_sB = self.satellites_totals()
        if N_s > 0:
            print "satellites"
            print "----------------------"
            print "overall fraction = %.3f" %(1.*N_s/N_tot)
            print "fraction without satellites in same halo = %.3f" %(1.*N_sA/N_s)
            print "fraction with satellites in same halo = %.3f" %(1.*N_sB/N_s)
            print
    #end info
    
    #---------------------------------------------------------------------------
    def random_subsample(self, N):
        """
        Set the indices defining a random subsample of size `N` 
        from the current galaxy sample
        """
        if N > self.total_galaxies:
            raise ValueError("Cannot select subsample of size %d from galaxy sample of size %d" %(N, self.total_galaxies))
        self.subsample_indices = sorted(np.random.choice(xrange(self.total_galaxies), N, replace=False))
    #end random_subsample
    
    #---------------------------------------------------------------------------
    def clear_subsample(self, N):
        """
        Clear the random subsample selection so we are using the full sample
        """
        self.subsample_indices = None
    #end clear_subsample
    
    #---------------------------------------------------------------------------    
    def restrict_halos(self, halo_condition):
        """
        Restrict the halos included in the current sample by inputing a boolean
        condition in string format, ``(key1 == value1)*(key2 == value2) + (key3 ==value3)``
        """
        self.restrictions.set_halo_flags(halo_condition)
        self._sample = self.restrictions.slice_frame(self._galaxies)
    #end restrict_halos
    
    #---------------------------------------------------------------------------
    def restrict_galaxies(self, galaxy_condition):
        """
        Restrict the galaxies included in the current sample by inputing a boolean
        condition in string format, ``(key1 == value1)*(key2 == value2) + (key3 ==value3)``
        """
        self.restrictions.set_galaxy_flags(galaxy_condition)
        self._sample = self.restrictions.slice_frame(self._galaxies)
        
        self._sample = self._add_halo_sizes(self._sample)
    #end restrict_galaxies
    
    #---------------------------------------------------------------------------
    def clear_restrictions(self):
        """
        Clear any restrictions
        """
        self.restrictions.clear_halo_flags()
        self.restrictions.clear_galaxy_flags()
        self._sample = None
    #end clear_restrictions
    #---------------------------------------------------------------------------
    def centrals_totals(self):
        """
        Return the total number of centrals, centrals without satellites, and
        centrals with satellites
        
        Returns
        -------
        N_c : int
            the total number of centrals
        N_cA : int
            the total number of centrals without satellites in the same halo
        N_cB : int
            the total number of centrals with satellites in the same halo
        """            
        # total centrals
        N_c = len(self.sample[self.sample.type == 'central'])

        # centrals with no sats
        N_cA = len(self.sample[(self.sample.type == 'central')&(self.sample.N_sat == 0)])

        # centrals with sats
        N_cB = len(self.sample[(self.sample.type == 'central')&(self.sample.N_sat > 0)])
        
        return N_c, N_cA, N_cB
    #end centrals_totals
    
    #---------------------------------------------------------------------------
    def satellites_totals(self):
        """
        Return the total number of satellites, satellites without other 
        satellites, and satellites with other satellites
        
        Returns
        -------
        N_s : int
            the total number of satellites
        N_sA : int
            the total number of satellites without satellites in the same halo
        N_sB : int
            the total number of satellites with satellites in the same halo
        """
        # total sats
        N_s = len(self.sample[self.sample.type == 'satellite'])

        # sats with no sats
        N_sA = len(self.sample[(self.sample.type == 'satellite')&(self.sample.N_sat == 1)])

        # sats with sats
        N_sB = len(self.sample[(self.sample.type == 'satellite')&(self.sample.N_sat > 1)])
        
        return N_s, N_sA, N_sB
    #end satellites_totals
          
    #---------------------------------------------------------------------------
    def load(self, filename, info_dict, halo_id, type_params, skip_lines=0):
        """
        Load objects from a file
        
        Parameters
        ----------
        filename : str
            The name of the file containing the info to load into each halo
        info_dict : dict
            A dictionary with keys corresponding to attributes of the 
            ``galaxy`` class and values equal to the column numbers to 
            read from the input file
        halo_id : str
            The name of the column defining the halo identifier
        type_params : dict
            A dictionary holding the parameters discerning centrals/satellites.
            The relevant keys are:
                ``column`` : str
                    The input column that separates centrals/satellites
                ``central`` : int
                    The value for centrals
                ``satellite`` : int
                    The value for satellites
        skip_lines : int, optional
            The number of lines to skip when reading the input file; default is 0
        """
        # save some column info for later
        self.halo_id = halo_id
        
        # read the data
        catalog_lines = open(filename, 'r').readlines()[skip_lines:]
        
        print "reading mock catalog..."
        bar = utilities.initializeProgressBar(len(catalog_lines))
        
        # now loop over each line, adding each galaxy to its halo
        data = []
        for i, line in enumerate(catalog_lines):

            bar.update(i+1)
            
            # the dictionary of galaxy info
            fields = line.split()
            row = {col_name : float(fields[index]) for col_name, index in info_dict.iteritems()}
            
            type_col = type_params.get('column', None)
            if type_col is not None:
                galaxy_type = row.pop(type_col)
                if galaxy_type == type_params['central']:
                    row['type'] = 'central'
                elif galaxy_type == type_params['satellite']:
                    row['type'] = 'satellite'
                else:
                    raise ValueError("Galaxy type not recognized")
                
            # create Series with the name equal to the row number
            series = pd.Series(row)
            data.append(series)
            
        # append to the DataFrame    
        index = pd.Index(range(len(data)), name='galaxy_id')
        self._galaxies = pd.DataFrame(data, index=index)
        
        # now add the N_cen, N_sat columns
        if type_col is not None:
            self._galaxies = self._add_halo_sizes(self._galaxies)
    #end load
    
    #---------------------------------------------------------------------------
    def _add_halo_sizes(self, df):
        """
        Compute N_cen per halo and add to the input DataFrame as a column
        """
        # delete N_sat, N_cen columns if they exist
        if 'N_sat' in df.columns: del df['N_sat']
        if 'N_cen' in df.columns: del df['N_cen']
        
        # these are grouped by type and then by halo id
        halos = df.groupby(['type', self.halo_id])
        
        # the sizes
        sizes = halos.size()
        
        # add N_cen if there are any centrals in this sample
        if 'central' in sizes.index.levels[0]:
            N_cen = pd.DataFrame(sizes['central'], columns=['N_cen'])
            df = df.join(N_cen, on=self.halo_id, how='left')
            
            # now fill missing values with zeros
            df.N_cen.fillna(value=0., inplace=True)
        
        # add the satellites
        if 'satellite' in sizes.index.levels[0]:
            N_sat = pd.DataFrame(sizes['satellite'], columns=['N_sat'])
            df = df.join(N_sat, on=self.halo_id, how='left')
    
            # now fill missing values with zeros
            df.N_sat.fillna(value=0., inplace=True)
        
        return df
    #end _add_halo_sizes
    
    #---------------------------------------------------------------------------
    def compute_collision_groups(self, radius, units, coord_keys=['x', 'y'], nprocs=1):
        """
        Compute fiber collision groups using the specified radius as the 
        collision scale. 
        
        Notes
        -----
        For each collision group found, this method adds a list of ``GalaxyID`` 
        named tuples corresponding to the halo id and galaxy id of each member
        
        Parameters
        ----------
        radius : float
            The value specifying the grouping radius to use when running the 
            Friends-of-Friends algorithm
        units : str, {`absolute`, `relative`, `degrees`}
            The units of the collision radius. Will be converted to match 
            `self.units`. If in `degrees`, the corresponding physical scale
            at `self.redshift` is computed
        coord_keys : list, optional
            The names of the attributes holding the coordinate values. Default
            is `[`x`, `y`]`
        nprocs : int, optional
            The number of processors to use when running the FOF algorithm. 
            Default is `1`
        """
        # first, check input value for radius units
        choices = ['absolute', 'relative', 'degrees']
        if units not in choices:
            raise ValueError("Invalid argument for `units`; must be one of %s" %choices)
            
        # intialize the empty group finder
        grp_finder = angularFOF.groupFinder(coord_keys=coord_keys, nprocs=nprocs)
        
        # add the galaxies, making copies
        grp_finder.addGalaxies(self._galaxies)
        
        # convert radius if we need to
        if units != self.units:
            
            if units == 'degrees':
                radius = evolution.physical_size(self.redshift, radius, params=self.cosmo)
                units = 'absolute'

            if self.units == 'absolute' and units == 'relative':
                radius *= h_conversion_factor('distance', units, self.units, self.cosmo['h'])
                units = 'absolute'
            elif self.units == 'relative' and units == 'absolute':
                radius *= h_conversion_factor('distance', units, self.units, self.cosmo['h']) 
                units = 'relative'
                
        if self.units != units:
            raise ValueError("Error converting collision radius to same units as coordinates")
            
        self.collision_radius = radius # should be in units of `self.units`
        self.coord_keys = coord_keys
           
        # now run the FOF algorithm
        groups = grp_finder.findGroups(self.collision_radius)
        
        # make a DataFrame of the group numbers
        group_numbers = {g.galaxy_id : gr_num for gr_num in groups for g in groups[gr_num].members}
        index = pd.Index(group_numbers.keys(), name='galaxy_id')
        group_df = pd.DataFrame(group_numbers.values(), index=index, columns=['group_number'])
        
        # now join to the main table
        self._galaxies = self._galaxies.join(group_df)
        
        del grp_finder   
    #end compute_collision_groups
    
    #---------------------------------------------------------------------------
    def collision_group(self, group_number):
        """
        Return a DataFrame of the member galaxies of the group specified 
        by the input group number. 
                
        Returns
        -------
        df : pandas.DataFrame
            a pandas DataFrame where each row is an object in this group
        """
        return self._galaxies[self._galaxies.group_number == group_number]
    #end collision_group
    
    #---------------------------------------------------------------------------
    def halo(self, halo_number):
        """
        Return a DataFrame of the member galaxies of the halo specified 
        by the input halo number. 
                
        Returns
        -------
        df : pandas.DataFrame
            a pandas DataFrame where each row is an object in this halo
        """
        return self._galaxies[self._galaxies[self.halo_id] == halo_number]
    #end collision_group
    
    #---------------------------------------------------------------------------     
    def _assign_fibers_multi(self, group):
        """
        Assign fibers to multi-galaxy collision groups by setting 
        ``collided = 1`` for the galaxy that collides with the most objects
        """
        # first shuffle the member ids, so we select random element when tied
        group_ids = list(group.index)
        np.random.shuffle(group_ids)
        
        collided_ids = []
        while len(group_ids) > 1:
           
            # compute the number of collisions
            dists = squareform(pdist(group[self.coord_keys], metric='euclidean'))
            n_collisions = np.sum((dists > 0.)&(dists <= self.collision_radius), axis=0)
                    
            # find the object that collides with the most
            collided_index = n_collisions.argmax()    
        
            # make the collided galaxy and remove from group
            collided_id = group_ids.pop(collided_index)
            group = group.drop(collided_id)
            
            # only make this a collided object if its n_collisions > 0
            # if n_collisions = 0, then the object can get a fiber for free
            if n_collisions[collided_index] > 0:
                collided_ids.append(collided_id)
        
        return collided_ids
    #end _assign_fibers_multi
        
    #---------------------------------------------------------------------------
    def assign_fibers(self, resolution_fraction):
        """
        Assign fibers to galaxies, attempting to mimic the SDSS tiling algorithm
        by maximizing the number of galaxies with fibers
        """
        if not 'group_number' in self._galaxies.columns:
            raise ValueError("Cannot assign fibers before computing collision groups")
            
        # first clear any old fiber assignments
        self.clear_fiber_assignments()
        
        # first, make a DataFrame with group_size as a column
        groups = self._galaxies.groupby('group_number')
        sizes = pd.DataFrame(groups.size(), columns=['group_size'])
        frame = self._galaxies.join(sizes, on='group_number', how='left')
        
        # handle group size = 2, setting collided = 1 for random element
        groups_pairs = frame[frame.group_size == 2].groupby('group_number')
        index = [np.random.choice(v) for v in groups_pairs.groups.values()]
        self._galaxies.loc[index, 'collided'] = np.ones(len(index))
        
        # handle group size > 2
        groups_multi = frame[frame.group_size > 2].groupby('group_number')
        collided_ids = []
        for group_number, group in groups_multi:
            collided_ids += self._assign_fibers_multi(group)
        self._galaxies.loc[collided_ids, 'collided'] = np.ones(len(collided_ids))
        
        # print out the fiber collision fraction    
        f_collision = 1.*self.potentially_collided/self.total_galaxies
        print "potential collision fraction = %.3f" %f_collision
        
        # now resolve any galaxies
        print "using resolution fraction = %.3f" %(resolution_fraction)
        self.resolution_fraction = resolution_fraction
        
        if self.resolution_fraction < 1.:
        
            # randomly select values for resolved attribute
            probs = [1.-self.resolution_fraction, self.resolution_fraction]
            new_resolved = np.random.choice([0, 1], size=self.potentially_collided, p=probs)
            
            # set the new resolved values
            self._galaxies.loc[self._galaxies.collided == 1, 'resolved'] = new_resolved
                
        f_collision = 1.*self.total_collided/self.total_galaxies
        print "final collision fraction = %.3f" %f_collision
        
    #end assign_fibers
    
    #---------------------------------------------------------------------------
    def clear_fiber_assignments(self):
        """
        Clear the fiber assignments
        """        
        self.resolution_fraction = 1. # all galaxies are uncollided
        self._galaxies['resolved'] = np.zeros(self.total_galaxies)
        self._galaxies['collided'] = np.zeros(self.total_galaxies)
        
    #end clear_fiber_assignments
    
    #---------------------------------------------------------------------------
    def write_coordinates(self, filename, fields, units=None, header=[], 
                            temporary=False, replace_with_nearest=None):
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
        replace_with_nearest : str, bool, optional
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
    #end write
    
    #---------------------------------------------------------------------------
    def _replace_with_nearest(self, cols, fields):
        """
        Find the nearest neighbor for all (collided and unresolved) galaxies
        and replace the collided object values with those of the 
        nearest neighbor on the sky, i.e. double-counting
        """        
        # get the galaxy ids of galaxy to replace and the neighbors
        gal_ids = self.collided_unresolved_ids
        NN_ids  = self.nearest_neighbor_ids
        
        # initialize the output
        output = self.sample.copy()
        if isinstance(cols, bool):
            output.loc[gal_ids, fields] = self._galaxies.loc[NN_ids, fields].values    
        else:
            output.loc[gal_ids, cols] = self._galaxies.loc[NN_ids, cols].values
        
        return output[fields]
    
    #end _replace_with_nearest    
    #---------------------------------------------------------------------------
    def save(self, filename):
        """
        Save the `MockHOD` instance as a HDF5 file to the filename specified
        
        Parameters
        ----------
        filename : str 
            the filename to output to
        """
        print "saving mock catalog..."
        
        # first write out the galaxies
        store = pd.HDFStore(filename, 'w')
        
        # store the galaxies DataFrame
        store['galaxies'] = self._galaxies
        
        # now also store the actual mock catalog
        self._galaxies = None
        self._sample = None
        store.get_storer('galaxies').attrs.my_attribute = self
        
        store.close()
    #end save
    
    #---------------------------------------------------------------------------
    def plot_hod(self, mass_col, mass_units="h^{-1} M_\odot"):
        """
        Plot the HOD distribution of the sample, showing N_cen and N_sat 
        vs halo mass
        """  
        # reindex by haloid
        frame = self.sample.drop_duplicates(cols=self.halo_id)
        frame = frame.set_index(self.halo_id)
          
        mass = frame[mass_col]
        N_cen = frame.N_cen
        N_sat = frame.N_sat
        
        x, y_cen, err_cen, w_cen = utilities.bin(mass, N_cen, nBins=10, log=True)
        x, y_sat, err_sat, w_sat = utilities.bin(mass, N_sat, nBins=10, log=True)

        pfy.errorbar(x, y_cen, err_cen/np.sqrt(w_cen), marker='.', ls='--', label='central galaxies')
        pfy.errorbar(x, y_sat, err_cen/np.sqrt(w_cen), marker='.', ls='--', label='satellite galaxies')
        pfy.loglog(x, y_cen+y_sat, marker='.', c='k', label='all galaxies')
        
        ax = pfy.gca()
        pfy.plt.subplots_adjust(bottom=0.15)  
        ax.xlabel.update(r'$M_\mathrm{halo} (%s)$' %mass_units, fontsize=16)
        ax.ylabel.update(r'$\langle N(M) \rangle$', fontsize=16)
        
        return ax
    #end plot_hod
    
    #---------------------------------------------------------------------------
    def _compute_mass_pdf(self, masses):
        """
        Internal function to compute mass pdf
        """
        M_min = np.amin(masses)
        M_max = np.amax(masses)

        # first plot and then we normalize
        mass_bins = np.logspace(np.log10(M_min), np.log10(M_max), 50)
        pdf, bins, patches = pfy.hist(masses, bins=mass_bins, color='k')
        bincenters = 0.5*(bins[1:] + bins[:-1])
        pfy.plt.cla()

        # transform N(M) into dN/dlnM
        widths = np.diff(bins)
        dlogM = np.diff(np.log10(bins))
        pdf = 1.*pdf/(dlogM*sum(pdf))

        return bins, pdf, widths
    #end _compute_mass_pdf
    
    #---------------------------------------------------------------------------
    def plot_mass_distribution(self, mass_col, mass_units="h^{-1} M_\odot"):
        """
        Plot the mass distribution
        """
        # halo masses for all halos
        mass_all = np.asarray(self.sample[mass_col])
        
        # halo masses for satellites
        mass_sat = np.asarray(self.sample[self.sample.type == 'satellite'][mass_col])
        
        # all galaxies
        bins1, pdf1, dM1 = self._compute_mass_pdf(mass_all)

        # satellites
        bins2, pdf2, dM2 = self._compute_mass_pdf(mass_sat)


        ax = pfy.gca()
        pfy.bar(bins1[:-1], pdf1, width=dM1, bottom=pdf1*0., color=ax.color_list[0], alpha=0.5, grid='y', label='all galaxies')
        pfy.bar(bins2[:-1], pdf2, width=dM2, bottom=pdf2*0., color=ax.color_list[1], alpha=0.5, grid='y', label='satellites')

        pfy.plt.subplots_adjust(bottom=0.15)
        ax.x_log_scale()
        ax.ylabel.update("dp / dlnM", fontsize=16)
        ax.xlabel.update(r"$M_\mathrm{halo} \ (%s)$" %mass_units, fontsize=16)
        ax.legend(loc=0)

        return ax
    #end plot_mass_distribution
        
    #---------------------------------------------------------------------------
#endclass MockHOD

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
    
    def __iter__(self):
        return iter(self.condition)
            
    #---------------------------------------------------------------------------
    def _parse_condition(self, str_condition):
        """
        Parse the input string condition
        """
        import pyparsing as pp
        
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
              
    def __str__(self):
        return self.string_condition

#-------------------------------------------------------------------------------
class Sample(object):
    """
    A class to handle sample restrictions
    """
    def __init__(self):
        self.halo_flags   = None
        self.galaxy_flags = None
    
    def __str__(self):
        return "%-15s %s\n%-15s %s\n" \
            %("halo flags:", self.halo_flags, "galaxy flags:", self.galaxy_flags)
        
    def is_clear(self):
        """
        Check if the sample has no restrictions
        """
        return (self.halo_flags is None) and (self.galaxy_flags is None)
    
    def clear_halo_flags(self):
        """
        Clear any halo restrictions
        """
        self.halo_flags = None
            
    def set_halo_flags(self, halo_cond):
        """
        Set the halo boolean conditions
        """
        self.halo_flags = SampleFlags(halo_cond)
    
    def clear_galaxy_flags(self):
        """
        Clear any galaxy restrictions
        """
        self.galaxy_flags = None
            
    def set_galaxy_flags(self, galaxy_cond):
        """
        Set the galaxy boolean conditions
        """
        self.galaxy_flags = SampleFlags(galaxy_cond)
    
    def slice_frame(self, frame):
        """
        Return a sliced DataFrame corresponding to the boolean conditions
        set by `self.galaxy_flags` and `self.halo_flags`
        """
        galaxy_cond = halo_cond = np.ones(len(frame), dtype='bool')
        if self.galaxy_flags is not None:
            galaxy_cond = self._valid(frame, self.galaxy_flags.condition)
        if self.halo_flags is not None:
            halo_cond = self._valid(frame, self.halo_flags.condition)
        
        return frame[(galaxy_cond)&(halo_cond)]
    
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
#endclass Sample

#-------------------------------------------------------------------------------
        
