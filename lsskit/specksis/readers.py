
#------------------------------------------------------------------------------
def read_1d_data(filename):
    from nbodykit import files, pkresult
    d, meta = files.ReadPower1DPlainText(filename)
    pk = pkresult.PkResult.from_dict(d, ['k', 'power', 'modes'], sum_only=['modes'], **meta)
    return pk
   
#------------------------------------------------------------------------------ 
def read_2d_data(filename):
    from nbodykit import files, pkmuresult
    d, meta = files.ReadPower2DPlainText(filename)
    pkmu = pkmuresult.PkmuResult.from_dict(d, sum_only=['modes'], **meta)
    return pkmu

#------------------------------------------------------------------------------    
def load_data(filename):
    
    readers = [read_1d_data, read_2d_data]
    for reader in readers:
        try:
            return reader(filename)
        except Exception as e:
            continue
    else:
        raise IOError("failure to load data from `%s`: %s" %(filename, str(e)))