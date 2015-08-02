"""
    lib.py
    lsskit

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : library of functions to call from the command-line using entry points
"""
from . import HODMock
from flipper import flipperDict

def compute_fiber_collisions(**kwargs):
    """
    Compute fiber collisions and assign fibers for a 
    ``catio.HODMock`` instance
    
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


