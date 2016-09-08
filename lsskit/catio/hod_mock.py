"""
    hod_mock.py
    lsskit.catio

    __author__ : Nick Hand
    __email__ : nhand@berkeley.edu
    __desc__ : subclass of `catio.MockCatalog` to hold an HOD galaxy mock catalog
"""
from . import angularFOF, mock_catalog, _utils, numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd
try:
    from scipy.spatial import cKDTree as KDTree
except:
    from scipy.spatial import KDTree


class HODMock(mock_catalog.MockCatalog):
    """
    Subclass of `catio.MockCatalog` to represent a galaxy HOD mock catalog
    """                    
    #---------------------------------------------------------------------------
    # read-ony properties
    #---------------------------------------------------------------------------
    @property
    def total_galaxies(self):
        """
        Total number of galaxies in the mock catalog
        """
        return self._sample_total

    @property
    def potentially_collided(self):
        """
        Total number of potentially collided galaxies in the current sample
        """
        return len(self.sample[self.sample.collided == 1])
    
    @property
    def total_collided(self):
        """
        Total number of collided galaxies in the current sample
        """
        cond = (self.sample.collided == 1)&(self.sample.resolved == 0)
        return len(self.sample[cond])
    
    @property
    def total_resolved(self):
        """
        Total number of total resolved, collided galaxies in the current sample
        """
        return len(self.sample[self.sample.resolved == 1])
    
    @property
    def total_halos(self):
        """
        Total number of halos in the mock catalog
        """
        return len(self.sample.groupby(self.halo_id).groups)

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
        
    @property
    def nearest_neighbor_ids(self):
        """
        Return an Index with the galaxy ids of the nearest neighbors of all
        collided and unresolved galaxies
        """
        try:
            return self._nearest_neighbor_ids
        except AttributeError:
            self._find_nearest_neighbors()
            return self._nearest_neighbor_ids
    
    @property
    def collided_unresolved_ids(self):
        """
        Return an Index with the galaxy ids of all collided and 
        unresolved galaxies
        """
        try:
            return self._collided_unresolved_ids
        except AttributeError:
            self._find_nearest_neighbors()
            return self._collided_unresolved_ids    
    
    #---------------------------------------------------------------------------
    # internal utility functions
    #---------------------------------------------------------------------------
    def _add_halo_sizes(self, df):
        """
        Internal function to compute number of centrals/satellites per halo and 
        add the information to the input DataFrame as a column
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
    
    def _find_nearest_neighbors(self):
        """
        Internal function to compute the nearest neighbors of all 
        collided/unresolved galaxies
        """
        # we can double count using any uncollided galaxies (for which
        # we have a redshift)
        cond = (self._data.collided == 0)|(self._data.resolved == 1)
        uncollided_gals = self._data[cond]
    
        # initialize the kdtree for NN calculations
        tree = KDTree(uncollided_gals[self.coord_keys])
    
        # find the NN for only the collided galaxies
        cond = (self.sample.collided == 1)&(self.sample.resolved == 0)
        collided_gals = self.sample[cond]
        dists, inds = tree.query(collided_gals[self.coord_keys], k=1)
    
        self._collided_unresolved_ids = collided_gals.index
        self._nearest_neighbor_ids    = uncollided_gals.iloc[inds].index
        self._metadata += ['_collided_unresolved_ids', '_nearest_neighbor_ids']
    
    def _assign_fibers_multi(self, group):
        """
        Internal function to assign fibers to multi-galaxy collision groups by setting 
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
    
    def _replace_with_nearest(self, cols, fields):
        """
        Internal function to find the nearest neighbor for all 
        (collided and unresolved) galaxies and replace the collided object 
        values with those of the nearest neighbor on the sky, i.e. double-counting
        """        
        # get the galaxy ids of galaxy to replace and the neighbors
        gal_ids = self.collided_unresolved_ids
        NN_ids  = self.nearest_neighbor_ids
        
        # initialize the output
        output = self.sample.copy()
        if isinstance(cols, bool):
            output.loc[gal_ids, fields] = self._data.loc[NN_ids, fields].values    
        else:
            output.loc[gal_ids, cols] = self._data.loc[NN_ids, cols].values
        
        return output[fields]
          
    #---------------------------------------------------------------------------
    # info functions
    #---------------------------------------------------------------------------
    def info(self):
        """
        Print out the sample info
        """
        N_tot = self.total_galaxies
        print("total number of galaxies = %d" %N_tot)
        print("total number of halos = %d" %self.total_halos)
        print() 
        
        N_c, N_cA, N_cB = self.centrals_totals()
        if N_c > 0:
            print("centrals")
            print("----------------------")
            print("overall fraction = %.3f" %(1.*N_c/N_tot))
            print("fraction without satellites in same halo = %.3f" %(1.*N_cA/N_c))
            print("fraction with satellites in same halo = %.3f" %(1.*N_cB/N_c))
            print() 
        
        N_s, N_sA, N_sB = self.satellites_totals()
        if N_s > 0:
            print("satellites")
            print("----------------------")
            print("overall fraction = %.3f" %(1.*N_s/N_tot))
            print("fraction without satellites in same halo = %.3f" %(1.*N_sA/N_s))
            print("fraction with satellites in same halo = %.3f" %(1.*N_sB/N_s))
            print()
          
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
          
    #---------------------------------------------------------------------------
    # general functions
    #---------------------------------------------------------------------------
    @classmethod
    def from_ascii(cls, filename, info_dict, halo_id, object_types, skip_lines=0, **meta):
        """
        Create a HODMock instance by reading object information from an 
        ASCII file
        
        Parameters
        ----------
        filename : str
            The name of the file containing the info to load into each halo
        info_dict : dict
            A dictionary with keys corresponding to the names of the columns
            in the `DataFrame` and the values corresponding to the column 
            numbers to read from the input file
        halo_id : str
            The name of the column defining the halo identifier        
        object_types : dict
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
        meta['skip_lines'] = skip_lines
        meta['object_types'] = object_types
        meta['halo_id'] = halo_id
        mock = mock_catalog.MockCatalog.from_ascii(filename, info_dict, **meta)
        
        # add the N_cen, N_sat columns
        if object_types.get('column', None) is not None:
            mock._data = self._add_halo_sizes(mock._data)
        
        return mock

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
        grp_finder.addGalaxies(self._data)
        
        # convert radius if we need to
        if units != self.units:
            
            if units == 'degrees':
                radius = (radius*np.pi/180.)*self.cosmo.Da_z(self.redshift)
                units = 'absolute'

            if self.units == 'absolute' and units == 'relative':
                radius *= _utils.h_conversion_factor('distance', units, self.units, self.cosmo['h'])
                units = 'absolute'
            elif self.units == 'relative' and units == 'absolute':
                radius *= _utils.h_conversion_factor('distance', units, self.units, self.cosmo['h']) 
                units = 'relative'
                
        if self.units != units:
            raise ValueError("Error converting collision radius to same units as coordinates")
            
        self.collision_radius = radius # should be in units of `self.units`
        self.coord_keys = coord_keys
        self._metadata += ['collision_radius', 'coord_keys']
           
        # now run the FOF algorithm
        groups = grp_finder.findGroups(self.collision_radius)
        
        # make a DataFrame of the group numbers
        group_numbers = {g.objid : gr_num for gr_num in groups for g in groups[gr_num].members}
        index = pd.Index(group_numbers.keys(), name='objid')
        group_df = pd.DataFrame(group_numbers.values(), index=index, columns=['group_number'])
        
        # now join to the main table
        self._data = self._data.join(group_df)
        
        del grp_finder   

    def collision_group(self, group_number):
        """
        Return a DataFrame of the member galaxies of the group specified 
        by the input group number. 
                
        Returns
        -------
        df : pandas.DataFrame
            a pandas DataFrame where each row is an object in this group
        """
        return self._data[self._data.group_number == group_number]
    
    def halo(self, halo_number):
        """
        Return a DataFrame of the member galaxies of the halo specified 
        by the input halo number. 
                
        Returns
        -------
        df : pandas.DataFrame
            a pandas DataFrame where each row is an object in this halo
        """
        return self._data[self._data[self.halo_id] == halo_number]
               
    def assign_fibers(self, resolution_fraction):
        """
        Assign fibers to galaxies, attempting to mimic the SDSS tiling algorithm
        by maximizing the number of galaxies with fibers
        """
        if not 'group_number' in self._data.columns:
            raise ValueError("Cannot assign fibers before computing collision groups")
            
        # first clear any old fiber assignments
        self.clear_fiber_assignments()
        
        # first, make a DataFrame with group_size as a column
        groups = self._data.groupby('group_number')
        sizes = pd.DataFrame(groups.size(), columns=['group_size'])
        frame = self._data.join(sizes, on='group_number', how='left')
        
        # handle group size = 2, setting collided = 1 for random element
        groups_pairs = frame[frame.group_size == 2].groupby('group_number')
        index = [np.random.choice(v) for v in groups_pairs.groups.values()]
        self._data.loc[index, 'collided'] = np.ones(len(index))
        
        # handle group size > 2
        groups_multi = frame[frame.group_size > 2].groupby('group_number')
        collided_ids = []
        for group_number, group in groups_multi:
            collided_ids += self._assign_fibers_multi(group)
        self._data.loc[collided_ids, 'collided'] = np.ones(len(collided_ids))
        
        # print out the fiber collision fraction    
        f_collision = 1.*self.potentially_collided/self.total_galaxies
        print("potential collision fraction = %.3f" %f_collision)
        
        # now resolve any galaxies
        print("using resolution fraction = %.3f" %(resolution_fraction))
        self.resolution_fraction = resolution_fraction
        self._metadata.append('resolution_fraction')
        
        if self.resolution_fraction < 1.:
        
            # randomly select values for resolved attribute
            probs = [1.-self.resolution_fraction, self.resolution_fraction]
            new_resolved = np.random.choice([0, 1], size=self.potentially_collided, p=probs)
            
            # set the new resolved values
            self._data.loc[self._data.collided == 1, 'resolved'] = new_resolved
                
        f_collision = 1.*self.total_collided/self.total_galaxies
        print("final collision fraction = %.3f" %f_collision)
        
    def clear_fiber_assignments(self):
        """
        Clear the fiber assignments
        """        
        self.resolution_fraction = 1. # all galaxies are uncollided
        self._data['resolved'] = np.zeros(self.total_galaxies)
        self._data['collided'] = np.zeros(self.total_galaxies)
        
    def restrict_halos(self, halo_condition):
        """
        Restrict the halos included in the current sample by inputing a boolean
        condition in string format, 
        
        ``(key1 == value1)*(key2 == value2) + (key3 ==value3)``
        """
        self.restrictions.set_flags('halo', halo_condition)
        self._sample = self.restrictions.slice_frame(self._data)
    
    def restrict_galaxies(self, galaxy_condition):
        """
        Restrict the galaxies included in the current sample by inputing a boolean
        condition in string format, 
        
        ``(key1 == value1)*(key2 == value2) + (key3 ==value3)``
        """
        self.restrictions.set_flags('galaxy', galaxy_condition)
        self._sample = self.restrictions.slice_frame(self._data)
        self._sample = self._add_halo_sizes(self._sample)
    
    #---------------------------------------------------------------------------
    # plotting functions
    #---------------------------------------------------------------------------
    def plot_hod(self, mass_col, N_bins=10):
        """
        Plot the HOD distribution of the sample, showing N_cen and N_sat 
        vs halo mass
        """  
        from matplotlib import pyplot as plt
        from .tools import bin
        
        # reindex by haloid
        frame = self.sample.drop_duplicates(self.halo_id)
        frame = frame.set_index(self.halo_id)
        
        # get the masses, N_cen, and N_sat  
        mass = frame[mass_col]
        N_cen = frame.N_cen
        N_sat = frame.N_sat
        
        # do the binning
        x, y_cen, err_cen, w_cen = bin(mass, N_cen, nBins=N_bins, log=True)
        x, y_sat, err_sat, w_sat = bin(mass, N_sat, nBins=N_bins, log=True)

        # plot
        plt.errorbar(x, y_cen, err_cen/np.sqrt(w_cen), marker='.', ls='--', label='central galaxies')
        plt.errorbar(x, y_sat, err_sat/np.sqrt(w_sat), marker='.', ls='--', label='satellite galaxies')
        plt.loglog(x, y_cen+y_sat, marker='.', c='k', label='all galaxies')
        
        ax = plt.gca()
        plt.subplots_adjust(bottom=0.15)  
        if self.units == 'relative':
            mass_units = "h^{-1} M_\odot"
        else:
            mass_units = "M_\odot"
        ax.set_xlabel(r'$M_\mathrm{halo} (%s)$' %mass_units, fontsize=16)
        ax.set_ylabel(r'$\langle N(M) \rangle$', fontsize=16)
        
        return ax
        
    def plot_mass_distribution(self, mass_col, mass_units="h^{-1} M_\odot"):
        """
        Plot the mass distribution of all galaxies, and satellites only
        """
        from matplotlib import pyplot as plt
        
        # halo masses for all halos
        mass_all = np.asarray(self.sample[mass_col])
        
        # halo masses for satellites
        mass_sat = np.asarray(self.sample[self.sample.type == 'satellite'][mass_col])
        
        # all galaxies and satellites only
        bins1, pdf1, dM1 = _utils.compute_pdf(mass_all, log=True)
    
        # plot
        ax = plt.gca()
        plt.bar(bins1[:-1], pdf1, width=dM1, bottom=pdf1*0., color=ax.next_color, alpha=0.5, grid='y')

        # make it look nice
        plt.subplots_adjust(bottom=0.15)
        ax.set_xscale('log')
        ax.set_ylabel("d$p$ / d $\mathrm{log_{10}}M$", fontsize=16)
        if self.units == 'relative':
            mass_units = "h^{-1} M_\odot"
        else:
            mass_units = "M_\odot"
        ax.set_xlabel(r"$M_\mathrm{halo} \ (%s)$" %mass_units, fontsize=16)
        ax.legend(loc=0)

        return ax        


        
