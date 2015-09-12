"""
    io.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : I/O tools for nbodykit's PkmuResult and PkResult
"""
from nbodykit import files, pkresult, pkmuresult, plugins
from . import tools
from .. import numpy as np


def get_Pshot(power):
    """
    Return the shot noise from a power spectrum instance
    """
    if hasattr(power, 'volume'):
        return power.volume / power.N1
    elif hasattr(power, 'box_size'):
        return power.box_size**3 / power.N1
    elif all(hasattr(power, x) for x in ['Lx', 'Ly', 'Lz']):
        return power.Lx*power.Ly*power.Lz / power.N1
    else:
        raise ValueError("cannot compute shot noise")
        
#------------------------------------------------------------------------------
# readers
#------------------------------------------------------------------------------
def read_1d_data(filename, columns=['k', 'power', 'modes']):
    """
    Read a `PkResult` from a file with the format used by ``nbodykit::power.py``
    """
    d, meta = files.ReadPower1DPlainText(filename)
    
    # try to extract the columns from the first line of the file
    cols = open(filename, 'r').readline()
    if cols[0] == '#':
        columns = cols.split()[1:]
    pk = pkresult.PkResult.from_dict(d, columns, sum_only=['modes'], **meta)
    return pk
   
def read_2d_data(filename, columns=None):
    """
    Read a `PkmuResult` from a file with the format used by ``nbodykit::power.py``
    """
    d, meta = files.ReadPower2DPlainText(filename)
    pkmu = pkmuresult.PkmuResult.from_dict(d, sum_only=['modes'], **meta)
    return pkmu
  
def load_data(filename, columns=['k', 'power', 'modes']):
    """
    Load either a ``PkmuResult`` or ``PkResult`` from file, assuming that the
    file has the format used by ``nbodykit::power.py``
    """
    readers = [read_1d_data, read_2d_data]
    for reader in readers:
        try:
            return reader(filename, columns)
        except Exception as e:
            continue
    else:
        raise IOError("failure to load data from `%s`: %s" %(filename, str(e)))

#------------------------------------------------------------------------------
# writers
#------------------------------------------------------------------------------ 
def write_plaintext(data, filename):
    """
    Write either a `PkResult` or ``PkmuResult`` instance as a plaintext file, using
    the format used by ``nbodykit::power.py``
    
    Parameters
    ----------
    data : nbodykit.PkmuResult, nbodykit.PkResult
        the power instance to write out
    filename : str
        the desired name for the output file
    """
    if isinstance(data, pkresult.PkResult):
        write_1d_plaintext(data, filename)
    elif isinstance(data, pkmuresult.PkmuResult):
        write_2d_plaintext(data, filename)
    else:
        raise ValueError("input power must be a `PkmuResult` or `PkResult` instance")
               
def write_2d_plaintext(power, filename):
    """
    Write a `PkmuResult` instance as a plaintext file, using the format used 
    by ``nbodykit::power.py``
    """
    # format the output
    result = {name:power.data[name].data for name in power.columns}
    result['edges'] = [power.kedges, power.muedges]
    meta = {k:getattr(power, k) for k in power._metadata}
    
    # and write
    storage = plugins.PowerSpectrumStorage.new('2d', filename)
    storage.write(result, **meta)
    
def write_1d_plaintext(power, filename):
    """
    Write a `PkResult` instance as a plaintext file, using the format used 
    by ``nbodykit::power.py``
    """
    # format the output
    result = [power.data[name].data for name in ['k', 'power', 'modes']]
    meta = {k:getattr(power, k) for k in power._metadata}
    meta['edges'] = power.kedges
    
    # and write
    storage = plugins.PowerSpectrumStorage.new('1d', filename)
    storage.write(result, **meta)
    
def write_analysis_file(filename, data, columns, subtract_shot_noise=True, 
                        kmin=None, kmax=None):
    """
    Write either a `PkResult``, ``PkmuResult``, or set of ``PkResult` objects
    representing multipoles,  as a plaintext file, with a format designed 
    for easy analysis 
    
    Notes
    -----
    The format is:
    Nk [Nmu|Nell]
    col1_name col2_name col3_name
    col1_0 col2_0 col3_0...
    col1_1 col2_1 col3_1...
    ...
    
    The difference between 1D and 2D results can be obtained by checking
    the size of the shape output on the first line
    
    Parameters
    ----------
    filename : str
        the desired name of the output file
    data : nbodykit.PkResult, nbodykit.PkmuResult, SpectraSet
        the power instance to write
    columns : list of str
        list of strings specifying the names of the columns to write to file
    subtract_shot_noise : bool, optional
        if `True`, subtract the shot noise before outputing to file. Default is `True`
    kmin : float or array_like
        the minimum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    kmax : float or array_like
        the maximum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    """
    # if a spectra set was passed, concatenate the data
    if not isinstance(data, (pkmuresult.PkmuResult, pkresult.PkResult)):
        if 'ell' not in data.dims:
            raise ValueError('SpectraSet passed but no `ell` dimension')
        tostack = []
        for ell in data['ell']:
            x = data.sel(ell=ell).values
            if subtract_shot_noise and ell == 0:
                x.data['power'] -= get_Pshot(x)
            tostack.append(x.data.copy())
        data = np.vstack(tostack).T
    # a PkmuResult or PkResult was passed
    else:
        if subtract_shot_noise: 
            data.data['power'] -= get_Pshot(data)
        data = data.data.copy()
    
    # get rid of the silly mask
    data = data.data
    
    # checks and balances
    if 'error' not in data.dtype.names:
        raise RuntimeError("probably not a good idea to write a data file with no errors")
    if not all(col in data.dtype.names for col in columns):
        args = (str(columns), str(data.dtype.names))
        raise RuntimeError("mismatch between desired columns %s and present columns %s" %args)

    # get the data
    data = tools.trim_and_align_data(data, kmin=kmin, kmax=kmax)
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
            
def write_power_analysis_file(filename, data, columns, subtract_shot_noise=True, 
                                kmin=None, kmax=None):
    """
    Write either a `PkResult``, ``PkmuResult`` as a plaintext file, with a 
    format designed for easy analysis 
    
    Notes
    -----
    The format is:
    Nk [Nmu]
    col1_name col2_name col3_name
    col1_0 col2_0 col3_0...
    col1_1 col2_1 col3_1...
    ...
    
    The difference between 1D and 2D results can be obtained by checking
    the size of the shape output on the first line
    
    Parameters
    ----------
    filename : str
        the desired name of the output file
    data : nbodykit.PkResult, nbodykit.PkmuResult
        the power instance to write
    columns : list of str
        list of strings specifying the names of the columns to write to file
    subtract_shot_noise : bool, optional
        if `True`, subtract the shot noise before outputing to file. Default is `True`
    kmin : float or array_like
        the minimum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    kmax : float or array_like
        the maximum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    """
    if subtract_shot_noise: 
        data.data['power'] -= get_Pshot(data)
    data = data.data.copy()
    
    # get rid of the silly mask
    data = data.data
    
    # checks and balances
    if 'error' not in data.dtype.names:
        raise RuntimeError("probably not a good idea to write a data file with no errors")
    if not all(col in data.dtype.names for col in columns):
        args = (str(columns), str(data.dtype.names))
        raise RuntimeError("mismatch between desired columns %s and present columns %s" %args)

    # get the data
    data = tools.trim_and_align_data(data, kmin=kmin, kmax=kmax)
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
            

def write_poles_analysis_file(filename, data, pkmu, columns, 
                                subtract_shot_noise=True, kmin=None, kmax=None):
    """
    Write a set of ``PkResult` objects representing multipoles, as a plaintext 
    file, with a format designed for easy analysis. The file also includes
    the mu values and weights needed to compute the theoretical multipoles
    properly
    
    Notes
    -----
    The format is:
    Nk Nell
    col1_name col2_name col3_name
    col1_0 col2_0 col3_0...
    col1_1 col2_1 col3_1...
    ...
    
    Parameters
    ----------
    filename : str
        the desired name of the output file
    data : SpectraSet
        the set of multipoles to write
    pkmu : nbodykit.PkmuResult
        the P(k,mu) instance which has the mu and weight values
    columns : list of str
        list of strings specifying the names of the columns to write to file
    subtract_shot_noise : bool, optional
        if `True`, subtract the shot noise before outputing to file. Default is `True`
    kmin : float or array_like
        the minimum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    kmax : float or array_like
        the maximum wavenumber in `h/Mpc` to consider. can specify a value
        for each mu bin, otherwise same value used for all mu bins
    """
    if 'ell' not in data.dims:
        raise ValueError('SpectraSet passed but no `ell` dimension')
    tostack = []
    for ell in data['ell']:
        x = data.sel(ell=ell).values
        if subtract_shot_noise and ell == 0:
            x.data['power'] -= get_Pshot(x)
        tostack.append(x.data.copy())
    data = np.vstack(tostack).T
    
    # get rid of the silly mask
    data = data.data
    
    # checks and balances
    if 'error' not in data.dtype.names:
        raise RuntimeError("probably not a good idea to write a data file with no errors")
    if not all(col in data.dtype.names for col in columns):
        args = (str(columns), str(data.dtype.names))
        raise RuntimeError("mismatch between desired columns %s and present columns %s" %args)

    # get the data
    data = tools.trim_and_align_data(data, kmin=kmin, kmax=kmax)
    shape = data.shape
    
    # write out the multipoles
    with open(filename, 'w') as ff:
        ff.write("{:d} {:d}\n".format(*shape))
        ff.write(" ".join(columns)+"\n")
        np.savetxt(ff, data[columns].ravel(order='F'))
        
    # now do the weights
    modes = pkmu['modes'].data
    mu = np.nan_to_num(pkmu['mu'].data)
    weights = modes/modes.sum(axis=-1)[:,None]
    data = np.empty(weights.shape, dtype=[('k', 'f8'), ('weights', 'f8'), ('mu', 'f8')])
    data['mu'] = mu; data['weights'] = weights; data['k'] = pkmu.index['k_center']
    data = tools.trim_and_align_data(data, kmin=kmin, kmax=kmax)
    
    # write out the weights
    with open(filename, 'a') as ff:
        shape = data.shape
        columns = ['mu', 'weights']
        ff.write("{:d} {:d}\n".format(*shape))
        ff.write(" ".join(columns)+"\n")
        np.savetxt(ff, data[columns].ravel(order='F'))
    
    
def read_analysis_file(filename):
    """
    Read an ``analysis file`` as output by ``write_analysis_file``
    """
    # read the data first
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
    
    # mu/weights?
    extra = None
    if len(lines) > N+2:
        try:
            shape = tuple(map(int, lines[N+2].split()))
            columns = lines[N+3].split()
            N2 = np.prod(shape)
            data2 = np.asarray([map(float, line.split()) for line in lines[N+4:N+4+N2]])
            
            dtype = [(col, 'f8') for col in columns]
            extra = np.empty(shape, dtype=dtype)
            for i, col in enumerate(columns):
                extra[col] = data2[...,i].reshape(shape, order='F')
        except Exception as e:
            raise RuntimeError("error parsing mu/weights: %s" %str(e))

    return toret if extra is None else (toret, extra)
            
            
        
            
