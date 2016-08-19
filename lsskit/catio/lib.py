"""
    lib.py
    lsskit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : library of functions to call from the command-line using entry points
"""
from . import HODMock, HaloMock
from flipper import flipperDict

def compute_fiber_collisions(**kwargs):
    """
    Compute fiber collisions and assign fibers for a 
    ``catio.HODMock`` instance
    
    Parameters
    ----------
    kwargs : key/value pairs
        The required keyword arguments are:
            param_file : str
                the name of the parameter file
            compute_groups : bool
                whether to compute the collisions groups
            assign_fibers : bool
                whether to assign fibers to the galaxies
    """
    params = flipperDict.flipperDict()
    params.readFromFile(kwargs['param_file'])
    
    # compute the collision groups
    if kwargs['compute_groups']:
    
        # load the mock
        mock = HODMock.from_hdf(params['mock_file'])
        
        print("computing collision groups...")
        mock.compute_collision_groups(params['radius'], 
                                        params['radius_units'], 
                                        coord_keys=params['coordinate_keys'], 
                                        nprocs=params['nprocs'])
                                                    
        
        mock.to_hdf(params['mock_file'])
    
    # assign fibers
    if kwargs['assign_fibers']:
        
        # load the mock
        mock = HODMock.from_hdf(params['mock_file'])
        
        print("assigning fibers...")
        mock.assign_fibers(params['resolution_fraction'])
                
        # now save the file
        mock.to_hdf(params['mock_file'])
        
def load_mock(**kwargs):
    """
    Load and save the either a `HaloMock` or `HODMock` instance
    
    Parameters
    ----------
    kwargs : key/value pairs
        The required keyword arguments are:
            param_file : str
                the name of the parameter file
            type : str, {'galaxy', 'halo'}
                the type of MockCatalog to save, either 'halo' or 'galaxy'
    """
    # read in the parameter file
    params = flipperDict.flipperDict()
    params.readFromFile(kwargs['param_file'])

    # get the metadata
    meta = {}
    meta['redshift']   = params['redshift']
    meta['box_size']   = params['box_size']
    meta['units']      = params['units']
    meta['cosmo']      = params['cosmology']
    meta['skip_lines'] = params['input']['skip_lines']
    
    p = params['input']
    if kwargs['type'] == 'halo':
        mock = HaloMock.from_ascii(p['file'], p['fields'], params['halo_id'], **meta)
    elif kwargs['type'] == 'galaxy':
        mock = HODMock.from_ascii(p['file'], p['fields'], params['halo_id'], params['type_params'], **meta)
    else:
        raise ValueError("do not understand `type` keyword")

    # now save the file
    mock.to_hdf(params['outfile'])
    
def write_coordinates(**kwargs):
    """
    Write out the formatted coordinates of a MockCatalog
    
    Parameters
    ----------
    kwargs : key/value pairs
        The required keyword arguments are:
            param_file : str
                the name of the parameter file
            type : str, {'galaxy', 'halo'}
                the type of MockCatalog to save, either 'halo' or 'galaxy'
    """
    # read in the parameter file
    params = flipperDict.flipperDict()
    params.readFromFile(kwargs['param_file'])

    # load the mock file
    if kwargs['type'] == 'galaxy':
        cls = HODMock
    elif kwargs['type'] == 'halo':
        cls = HaloMock
    else:
        raise ValueError("do not understand `type` keyword")
    mock = cls.from_hdf(params['mock_file'])
 
    # clear any restrictions
    mock.clear_restrictions()

    # set the restrictions
    if params['galaxy_restrict'] is not None:
        mock.restrict_galaxies(params['galaxy_restrict'])

    if params['halo_restrict'] is not None:
        mock.restrict_halos(params['halo_restrict'])

    # write out the mock coordinates
    mock.to_coordinates(params['output_file'], params['output_fields'], 
                        units=params['output_units'], 
                        header=params['header'], 
                        replace_with_nearest=params.get('replace_with_nearest', False),
                        temporary=False)
                        
def gal_to_halo_samples(**kwargs):
    """
    Output halo samples matching the mass pdf of galaxy samples
    """
    import pandas as pd
    import numpy as np
    
    # load
    gals = HODMock.from_hdf(kwargs['galaxy_file'])
    halos = HaloMock.from_hdf(kwargs['halo_file'])

    # get cenA masses
    gals.restrict_galaxies("type == central")
    gals.restrict_halos("N_sat == 0")
    cenA_masses = gals.sample.mass

    # and cenB masses
    gals.clear_restrictions()
    gals.restrict_galaxies("type == central")
    gals.restrict_halos("N_sat > 0")
    cenB_masses = gals.sample.mass

    # get indices of "cenB" halos
    halos.restrict_by_mass_pdf(cenB_masses)
    cenB_inds = halos.sample.index

    # now do cenB from all other halos
    halos.clear_restrictions()
    other_inds = halos.sample.index.drop(cenB_inds)
    halos.restrict_by_index(other_inds)
    halos.restrict_by_mass_pdf(cenA_masses)
    cenA_inds = halos.sample.index

    # now do satA and satB
    gals.clear_restrictions()
    index = cenB_inds.copy()

    satA_inds, satB_inds = [], []
    for N_sat in gals.sample.N_sat.unique()[::-1]:
        if N_sat < 1: continue

        gals.clear_restrictions()
        gals.restrict_halos("N_sat == %d" %N_sat)
        gals.restrict_galaxies("type == satellite")
        masses = gals.sample.groupby('haloid').mass.first()
    
        halos.clear_restrictions()
        halos.restrict_by_index(index)
    
        if len(masses) < 500:
            halo_masses = halos.sample.mass.copy()
            def find_nearest(halo_masses, m):
                return abs(halo_masses-m).argmin()
            this_inds = []
            for m in masses:
                this_inds.append(find_nearest(halo_masses, m))
                halo_masses = halo_masses.drop(this_inds[-1])
        else:
            halos.restrict_by_mass_pdf(masses)
            this_inds = list(halos.sample.index)
        
        if N_sat == 1:
            satA_inds += list(np.repeat(this_inds, N_sat)) 
        else:
            satB_inds += list(np.repeat(this_inds, N_sat)) 
        index = index.drop(this_inds)
        
    satA_inds = pd.Index(satA_inds)
    satB_inds = pd.Index(satB_inds)

    halos.clear_restrictions()
    copy = halos.sample.copy()
    out = pd.DataFrame()
    for i, inds in enumerate([cenA_inds, cenB_inds, satA_inds, satB_inds]):
        this = copy.loc[inds].copy()
        this.loc[:, 'keep'] = True
        if i < 2: 
            this.loc[:, 'type'] = 'central'
        else:
            this.loc[:, 'type'] = 'satellite'
        if i % 2 == 0:
            this.loc[:, 'subtype'] = 'A'  
        else:
            this.loc[:, 'subtype'] = 'B'  
        out = out.append(this)
    out = out.loc[out.keep == True]
    del out['keep']

    out.to_hdf(kwargs['output'], 'data')




