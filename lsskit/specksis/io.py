"""
    io.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : I/O tools for nbodykit's PkmuResult and PkResult
"""
from nbodykit import files
from nbodykit.dataset import DataSet, Corr1dDataSet, Corr2dDataSet
from nbodykit.dataset import Power1dDataSet, Power2dDataSet 
from nbodykit.storage import MeasurementStorage

from . import tools
from .. import numpy as np
import os


def get_Pshot(power):
    """
    Return the shot noise from a power spectrum `DataSet`, 
    trying to extract it from the `attrs` attribute
    """
    if not hasattr(power, 'attrs'):
        raise ValueError('input power object in get_Pshot needs a `attrs` attribute')

    attrs = power.attrs
    volume = 0.
    if 'volume' in attrs:
        volume = attrs['volume']
    elif 'box_size' in attrs:
        volume = attrs['box_size']**3
    elif all(x in attrs for x in ['Lx', 'Ly', 'Lz']):
        volume = attrs['Lx']*attrs['Ly']*attrs['Lz']
    else:
        raise ValueError("cannot compute volume for shot noise")
    return volume / attrs['N1']
    
        
#------------------------------------------------------------------------------
# readers
#------------------------------------------------------------------------------
def read_cutsky_power_poles(filename, skiprows=31, **kwargs):
    """
    Return a list of `Power1dDataSet` objects for each multipole in
    the input data file
    """
    data = np.loadtxt(filename, skiprows=skiprows, comments=None)
    
    # make the edges
    dk = 0.005
    lower = data[:,0]-dk/2.
    upper = data[:,0]+dk/2.
    edges = np.array(zip(lower, upper))
    edges = np.concatenate([edges.ravel()[::2], [edges[-1,-1]]])

    toret = []
    columns = ['k', 'mono', 'quad', 'hexadec', 'modes']
    meta = {'edges':edges, 'sum_only':['modes'], 'force_index_match':True}
    d = data[:,[1, 2, 3, 4, -1]]
    
    return Power1dDataSet.from_nbkit(d, meta, columns=columns, **kwargs)
    
    
def load_momentum(filename, ell, ell_prime, **kwargs):
    """
    Return a list of mu powers as measured from momentum correlators
    """
    allowed = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (0, 4)]
    moments = (ell, ell_prime)
    if moments not in allowed:
        raise ValueError("(ell, ell_prime) must be one of %s" %str(allowed))
    
    # load the multipoles
    poles = Power1dDataSet.from_nbkit(*files.Read1DPlainText(filename), **kwargs)
    k = poles['k']
    power = []
    
    # P00
    if moments == (0, 0):
        P_mu0 = poles['power']
        power += [P_mu0, np.nan, np.nan, np.nan, np.nan]
    
    # P01
    elif moments == (0, 1):
        P_mu2 = k * poles['power_1.imag']
        power += [np.nan, P_mu2, np.nan, np.nan, np.nan]
        
    # P11 or P02
    elif moments == (1, 1) or moments == (0, 2):
        P0 = poles['power_0.real']
        P2 = poles['power_2.real']

        P_mu2 = k**2 * (P0 - 0.5 * P2)
        P_mu4 = k**2 * 1.5 * P2
        power += [np.nan, P_mu2, P_mu4, np.nan, np.nan]
    
    # P12 or P03
    elif moments == (1, 2) or moments == (0, 3):
        
        P1 = poles['power_1.imag']
        P3 = poles['power_3.imag']
            
        P_mu4 = k**3 * (P1 - 1.5 * P3)
        P_mu6 = k**3 * 5./2 * P3
        power += [np.nan, np.nan, P_mu4, P_mu6, np.nan]
        
    # P22 or P13 or P04
    else:
        
        P0 = poles['power_0.real']
        P2 = poles['power_2.real']
        P4 = poles['power_4.real']
        
        P_mu4 = k**4 * (P0 - 0.5*P2 + 3./8*P4)
        P_mu6 = k**4 * (1.5*P2 - 15./4*P4)
        P_mu8 = k**4 * 35./8 * P4
        power += [np.nan, np.nan, P_mu4, P_mu6, P_mu8]
        
    usecols = ['k', 'modes']
    new_poles = []
    for P in power:
        if np.isscalar(P) and np.isnan(P):
            new_poles.append(np.nan)
        else:
            copy = poles.copy()
            copy = copy[usecols]
            copy['power'] = P
            new_poles.append(copy)
    
    return new_poles
    
            
def load_correlation(filename, mode, usecols=[], mapcols={}, **kwargs):
    """
    Load a 1D or 2D correlation measurement and return a `DataSet`
    """
    if mode not in ['1d', '2d']:
        raise ValueError("`mode` must be on of '1d' or '2d'")

    if mode == '1d':
        toret = Corr1dDataSet.from_nbkit(*files.Read1DPlainText(filename), **kwargs)
    else:
        toret = Corr2dDataSet.from_nbkit(*files.Read2DPlainText(filename), **kwargs)
    
    # rename any variables
    if len(mapcols):
        for old_name in mapcols:
            toret.rename_variable(old_name, mapcols[old_name])
    
    # only return certain variables
    if len(usecols):
        toret = toret[usecols]

    return toret
    
def load_power(filename, mode, usecols=[], mapcols={}, **kwargs):
    """
    Load a 1D or 2D power measurement and return a `DataSet`
    """
    if mode not in ['1d', '2d']:
        raise ValueError("`mode` must be on of '1d' or '2d'")

    if mode == '1d':
        toret = Power1dDataSet.from_nbkit(*files.Read1DPlainText(filename), **kwargs)
    else:
        toret = Power2dDataSet.from_nbkit(*files.Read2DPlainText(filename), **kwargs)
    
    # rename any variables
    if len(mapcols):
        for old_name in mapcols:
            toret.rename_variable(old_name, mapcols[old_name])
    
    # only return certain variables
    if len(usecols):
        toret = toret[usecols]

    return toret


#------------------------------------------------------------------------------
# writers
#------------------------------------------------------------------------------ 
def write_plaintext(data, filename):
    """
    Write a 1D or 2D `DataSet` instance as a plaintext file
    
    Parameters
    ----------
    data : `DataSet`
        the data set instance to write out
    filename : str
        the desired name for the output file
    """
    if not isinstance(data, DataSet):
        raise TypeError("input `data` must be an instance of `DataSet`")
    
    if len(data.dims) == 1:
        mode = '1d'
    elif len(data.dims) == 2:
        mode = '2d'
    else:
        raise ValueError("`data` has too many dimensions to write to plaintext file")
               
    # format the output
    columns = [data[name] for name in data.variables]
    edges = [data.edges[dim] for dim in data.dims]
    if len(edges) == 1:
        edges = edges[0]
    
    # and write
    storage = MeasurementStorage.create(mode, filename)
    storage.write(edges, data.variables, columns, **data.attrs)
                
#------------------------------------------------------------------------------
# read/write analysis files
#------------------------------------------------------------------------------ 
def _write_analysis_file(filename, data, columns, coords, kmin, kmax):
    """
    Internal helper function to write out analysis file
    """
    # checks and balances
    if not all(col in data.dtype.names for col in columns):
        args = (str(columns), str(data.dtype.names))
        raise RuntimeError("mismatch between desired columns %s and present columns %s" %args)

    # get the data
    _, data = tools.trim_and_align_data(coords, data, kmin=kmin, kmax=kmax)
    data = np.squeeze(data)
    shape = data.shape
    
    # now output
    with open(filename, 'w') as ff:
        if len(shape) > 1:
            ff.write("{:d} {:d}\n".format(*shape))
            ff.write(" ".join(columns)+"\n")
            np.savetxt(ff, data[columns].ravel(order='F'))
        else:
            ff.write("{:d}\n".format(*shape))
            ff.write(" ".join(columns)+"\n")
            np.savetxt(ff, data[columns])
            
def write_analysis_file(kind, 
                        filename, 
                        power, 
                        columns, 
                        subtract_shot_noise=True, 
                        kmin=-np.inf, 
                        kmax=np.inf):
    """
    Write either P(k), P(k,mu) or stacked P_\ell(k) results
    to a plaintext file, with formatted designed to faciliate
    analysis
    
    Notes
    -----
    The format is:
    Nk [Nmu|Nell]
    col1_name col2_name col3_name
    col1_0 col2_0 col3_0...
    col1_1 col2_1 col3_1...
    ...
    
    The difference between 1D and 2D results can be obtained by 
    checking the size of the shape output on the first line
    
    Parameters
    ----------
    kind : {'power', 'poles'}
        either write out a P(k), P(k,mu) spectrum or multipoles
    filename : str
        the desired name of the output file
    power : {nbodykit.PkResult, nbodykit.PkmuResult, SpectraSet}
        the power instance to write
    columns : list of str
        list of strings specifying the names of the columns to write to 
        file
    subtract_shot_noise : bool, optional (`True`)
        if `True`, subtract the shot noise before outputing to file.
    kmin : {float, array_like}, optional
        the minimum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    kmax : {float, array_like}, optional
        the maximum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    """
    # case of P(k) or P(k,mu)
    if kind == 'power':
        Pshot = 0. if not subtract_shot_noise else get_Pshot(power)
        data = power.data.data.copy() # removes the mask
        data['power'] -= Pshot
        coords = [power.k_center, power.mu_center]
    
    # case of multipoles
    elif kind == 'poles':
        if 'ell' not in power.dims:
            raise ValueError('multipoles ``SpectraSet`` passed but no `ell` dimension')
    
        # stack the multipoles
        data = tools.stack_multipoles(power)
        
        # subtract shot noise from monopole
        p = power[0].values
        ells = list(power['ell'].values)
        if 0 in ells and subtract_shot_noise: 
            data['power'][:,ells.index(0)] -= get_Pshot(p)
        coords = [p.k_center, np.array(ells, dtype=float)]
    else:
        raise ValueError("first argument to `write_analysis_file` must be `power` or `poles`")    

    # do the common work
    _write_analysis_file(filename, data, columns, coords, kmin, kmax)


def read_analysis_file(filename):
    """
    Read an ``analysis file`` as output by ``write_analysis_file``
    """
    # read the data
    lines = open(filename, 'r').readlines()
    shape = tuple(map(int, lines[0].split()))
    columns = lines[1].split()
    N = np.prod(shape)
    data = np.asarray([map(float, line.split()) for line in lines[2:N+2]])
    
    # return a structured array
    dtype = [(col, 'f8') for col in columns]
    toret = np.empty(shape, dtype=dtype)
    for i, col in enumerate(columns):
        toret[col] = data[...,i].reshape(shape, order='F')
    return toret

