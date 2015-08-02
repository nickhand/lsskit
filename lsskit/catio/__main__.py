"""
    __main__.py
    lsskit.catio

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : functions to serve as entry points for console scripts
"""
from . import lib
import argparse as ap

class CustomFormatter(ap.ArgumentDefaultsHelpFormatter, ap.RawTextHelpFormatter):
    pass
        

def compute_fiber_collisions():
    """
    Compute fiber collisions and assign fibers for a 
    ``catio.HODMock`` instance
    """
    # initialize the parser
    desc = "load a mock catalog from file and compute the fiber collisions"
    parser = ap.ArgumentParser(description=desc)
    parser.formatter_class = CustomFormatter
    
    # add the arguments
    h = """the name of the parameter file, which should have the following parameters:
            mock_file : str
                the name of the mock catalog file to load
            radius : float
                the value specifying the grouping radius to use when running the 
                friends-of-friends algorithm
            radius_units : str, {`absolute`, `relative`, `degrees`}
                the units of the collision radius. Will be converted to match 
                the units of the mock catalog. If in `degrees`, the corresponding 
                physical scale at the redshift of the mock catalog is used
            coordinate_keys : list of str
                list of columns specifying the coordinates to use for grouping
            nprocs : int
                the number of processors to use when running the FoF algorithm
            resolution_fraction : float
                the number of collided galaxies to randomly "resolve", i.e., set
                them as no longer collided"""
    parser.add_argument('param_file', type=str, help=h)
    h = 'whether to compute the collision groups'
    parser.add_argument('--compute_groups', action='store_true', default=False, help=h)
    h = 'whether to assign fibers'
    parser.add_argument('--assign_fibers', action='store_true', default=False, help=h)

    lib.compute_fiber_collisions(**vars(parser.parse_args()))


def load_mock(args):
    """
    Load and save the `HaloMock` or `HODMock` instance
    """
    # setup the main parser
    desc = "load a mock catalog from an ASCII file"
    parser = ap.ArgumentParser(description=desc)
    parser.formatter_class = CustomFormatter
    
    h = """the name of the parameter file, which should have the following parameters:
            input : dict
                a dictionary with the following keys:
                    file : str
                        the name of the ASCII file
                    fields : dict
                        dictionary with keys giving the desired column names, and
                        values giving the corresponding column numbers in the 
                        ASCII file
                    skip_lines : int
                        the number of lines to skip in the ASCII file
            halo_id : str
                name of the column giving the halo ID data
            type_params : dict
                a dictionary used to distinguish between objects of different types. 
                The `column` key should give the name of the column holding the 
                type information and the `types` key holds a dict of the different types
            box_size : float
                the simulation box size
            redshift : float
                the simulaton redshift
            cosmology : str
                the name of the cosmological parameter file
            units : str, {'absolute', 'relative'}
                the type of units"""
    parser.add_argument('param_file', type=str, help=h)

    h = 'either load and save a `HaloMock` or `HODMock` class'
    parser.add_argument('--type', type=str, choices=['halo', 'galaxy'], required=True, help=h)

    lib.load_mock(**vars(parser.parse_args()))

