"""
    io.py
    lsskit.specksis

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : I/O tools for nbodykit's PkmuResult and PkResult
"""
from nbodykit import files, pkresult, pkmuresult, plugins
from .. import numpy as np

#------------------------------------------------------------------------------
# readers
#------------------------------------------------------------------------------
def read_1d_data(filename):
    """
    Read a `PkResult` from a file with the format used by ``nbodykit::power.py``
    """
    d, meta = files.ReadPower1DPlainText(filename)
    pk = pkresult.PkResult.from_dict(d, ['k', 'power', 'modes'], sum_only=['modes'], **meta)
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
    storage = plugins.PowerSpectrumStorage.get('2d', filename)
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
    storage = plugins.PowerSpectrumStorage.get('1d', filename)
    storage.write(result, **meta)
    
def write_analysis_file(filename, data, columns, remove_missing=True, 
                        subtract_shot_noise=True, reindex={}):
    """
    Write either a ``PkResult`` or ``PkmuResult`` as a plaintext file,
    with a format designed for easy analysis 
    
    Notes
    -----
    The format is:
    Nk [Nmu]
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
    remove_missing : bool, optional
        if `True`, remove any masked elements before writing. Default is `True`.
    subtract_shot_noise : bool, optional
        if `True`, subtract the shot noise before outputing to file. Default is `True`
    reindex : dict, optional
        dictionary with optional keys `k`, `mu`, specifying new bins to use for
        that dimension
    """
    # checks and balances
    if 'error' not in data:
        raise RuntimeError("probably not a good idea to write a data file with no errors")
    if subtract_shot_noise and (not hasattr(data, 'box_size') or not hasattr(data, 'N1')):
        raise RuntimeError("can't subtract shot noise without ``box_size`` and ``N1`` attributes")
    
    # reindex to different bins?
    if len(reindex):
        if 'k' in reindex:
            data = data.reindex_k(reindex['k'])
        if 'mu' in reindex:
            data = data.reindex_mu(reindex['mu'])
    
    # optional values
    Pshot = 0
    if subtract_shot_noise:
        Pshot = data.box_size**3 / data.N1
    valid = np.ones(data.data.shape, dtype=bool)
    if remove_missing:
        valid = ~data['power'].mask
    
    # now output
    with open(filename, 'w') as ff:
        if isinstance(data, pkmuresult.PkmuResult):
            towrite = map(np.ravel, [data[col][valid] for col in columns])
            towrite[columns.index('power')] -= Pshot
            
            ff.write("{:d} {:d}\n".format(*data.data.shape))
            ff.write(" ".join(columns)+"\n")
            np.savetxt(ff, zip(*towrite))
        elif isinstance(data, pkresult.PkResult):
            ff.write("{:d}\n".format(*data.data.shape))
            np.savetxt(ff, zip(*[data[col][valid] for col in columns]))
            
            
        
            
