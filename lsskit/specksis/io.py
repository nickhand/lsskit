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
def read_1d_data(filename):
    """
    Read a `PkResult` from a file with the format used by ``nbodykit::power.py``
    """
    d, meta = files.ReadPower1DPlainText(filename)
    
    # try to extract the columns from the first line of the file
    columns = open(filename, 'r').readline()
    if columns[0] == '#':
        columns = columns.split()[1:]
    else:
        columns = ['k', 'power', 'modes']
    pk = pkresult.PkResult.from_dict(d, columns, sum_only=['modes'], **meta)
    return pk
   
def read_2d_data(filename):
    """
    Read a `PkmuResult` from a file with the format used by ``nbodykit::power.py``
    """
    d, meta = files.ReadPower2DPlainText(filename)
    pkmu = pkmuresult.PkmuResult.from_dict(d, sum_only=['modes'], **meta)
    return pkmu
  
def load_data(filename):
    """
    Load either a ``PkmuResult`` or ``PkResult`` from file, assuming that the
    file has the format used by ``nbodykit::power.py``
    """
    readers = [read_1d_data, read_2d_data]
    for reader in readers:
        try:
            return reader(filename)
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
    Write either a (set of) `PkResult`` or ``PkmuResult`` as a plaintext file,
    with a format designed for easy analysis 
    
    Notes
    -----
    The format is:
    Nk [Nmu|Nell]
    col1_name col2_name col3_name
    col1_0 col2_0 col3_0...
    col1_1 col2_1 col3_1...
    ...
    
    The difference between P(k,mu) and P(k) results can be obtained by checking
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
    # checks and balances
    if 'error' not in data:
        raise RuntimeError("probably not a good idea to write a data file with no errors")
    
    # subtract shot noise?
    Pshot = 0
    if subtract_shot_noise: Pshot = get_Pshot(data)
    
    # get the data
    data = data.data.copy()
    data = tools.get_valid_data(data, kmin=kmin, kmax=kmax)
    shape = data.shape
    data['power'] -= Pshot
    
    # now output
    with open(filename, 'w') as ff:
        if len(shape) > 1:
            towrite = map(np.ravel, [data[col] for col in columns])
            ff.write("{:d} {:d}\n".format(*shape))
            ff.write(" ".join(columns)+"\n")
            np.savetxt(ff, zip(*towrite))
        else:
            ff.write("{:d}\n".format(*shape))
            ff.write(" ".join(columns)+"\n")
            np.savetxt(ff, zip(*[data[col] for col in columns]))
            
def read_analysis_file(filename):
    """
    Read an ``analysis file`` as output by ``write_analysis_file``
    """
    with open(filename, 'r') as ff:
        shape = tuple(map(int, ff.readline().split()))
        columns = ff.readline().split()
        data = np.loadtxt(ff)
    dtype = [(col, 'f8') for col in columns]
    toret = np.empty(shape, dtype=dtype)
    for i, col in enumerate(columns):
        toret[col] = data[...,i].reshape(shape)
        
    return toret
            
            
        
            
