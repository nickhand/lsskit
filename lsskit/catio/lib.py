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
        
        print "computing collision groups..."
        mock.compute_collision_groups(params['radius'], 
                                        params['radius_units'], 
                                        coord_keys=params['coordinate_keys'], 
                                        nprocs=params['nprocs'])
                                                    
        
        mock.to_hdf(params['mock_file'])
    
    # assign fibers
    if kwargs['assign_fibers']:
        
        # load the mock
        mock = HODMock.from_hdf(params['mock_file'])
        
        print "assigning fibers..."
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




