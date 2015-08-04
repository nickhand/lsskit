
def read_1d_data(filename):
    from nbodykit import files, pkresult
    d, meta = files.ReadPower1DPlainText(filename)
    pk = pkresult.PkResult.from_dict(d, ['k', 'power', 'modes'], sum_only=['modes'], **meta)
    return pk
   
def read_2d_data(filename):
    from nbodykit import files, pkmuresult
    d, meta = files.ReadPower2DPlainText(filename)
    pkmu = pkmuresult.PkmuResult.from_dict(d, sum_only=['modes'], **meta)
    return pkmu
  
def load_data(filename):
    
    readers = [read_1d_data, read_2d_data]
    for reader in readers:
        try:
            return reader(filename)
        except Exception as e:
            continue
    else:
        raise IOError("failure to load data from `%s`: %s" %(filename, str(e)))
        
def write_2d_plaintext(power, filename):
    from nbodykit import plugins

    # format the output
    result = {name:power.data[name].data for name in power.columns}
    result['edges'] = [power.kedges, power.muedges]
    meta = {k:getattr(power, k) for k in power._metadata}
    
    # and write
    storage = plugins.PowerSpectrumStorage.get('2d', filename)
    storage.write(result, **meta)
    
def write_1d_plaintext(power, filename):
    from nbodykit import plugins

    # format the output
    result = [power.data[name].data for name in ['k', 'power', 'modes']]
    meta = {k:getattr(power, k) for k in power._metadata}
    meta['edges'] = power.kedges
    
    # and write
    storage = plugins.PowerSpectrumStorage.get('1d', filename)
    storage.write(result, **meta)
    
def write_plaintext(data, filename):
    from nbodykit import pkresult, pkmuresult
    if isinstance(data, pkresult.PkResult):
        write_1d_plaintext(data, filename)
    elif isinstance(data, pkmuresult.PkmuResult):
        write_2d_plaintext(data, filename)
    else:
        raise ValueError("input power must be a `PkmuResult` or `PkResult` instance")