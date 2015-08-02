"""
    __main__.py
    lsskit.catio

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : functions to serve as entry points for console scripts
"""
from . import lib
import argparse


def compute_fiber_collisions():
    """
    Compute fiber collisions and assign fibers for a 
    ``catio.HODMock`` instance
    """
    # initialize the parser
    desc = "load a mock catalog from file and compute the fiber collisions"
    parser = argparse.ArgumentParser(description=desc)
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter
    
    # add the arguments
    parser.add_argument('param_file', type=str, help="the parameter file")
    h = 'whether to compute the collision groups'
    parser.add_argument('--compute_groups', action='store_true', default=False, help=h)
    h = 'whether to assign fibers'
    parser.add_argument('--assign_fibers', action='store_true', default=False, help=h)

    lib.compute_fiber_collisions(**vars(parser.parse_args()))

