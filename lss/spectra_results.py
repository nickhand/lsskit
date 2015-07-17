from glob import glob
import os
import itertools
import numpy as np
import pickle

#------------------------------------------------------------------------------
def enum_files(result_dir, basename, index):
    
    # get abs paths to directories
    result_dir = os.path.abspath(result_dir)
    
    # try to find all files
    for i, args in enumerate(index.iterargs()):
        try:            
            # see if path exists
            filename = basename.format(**args)
            f = os.path.join(result_dir, filename)
            if not os.path.exists(f): raise
        
            # yield key and filename
            yield index[i], f
        
        except:
            message = 'no file found for `%s`\n in directory `%s`' %(basename, result_dir)
            raise IOError(message)

#------------------------------------------------------------------------------
def read_1d_data(filename):
    from nbodykit import files, pkresult
    d, meta = files.ReadPower1DPlainText(filename)
    pk = pkresult.PkResult.from_dict(d, sum_only=['modes'], **meta)
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
 
#------------------------------------------------------------------------------        
class SpectraIndex:
    
    def __init__(self, levels, names):
        
        self.levels = levels
        self.names = names
        dtype = np.dtype([(name, np.object) for name in names])
        self.values = np.array(list(itertools.product(*self.levels)), dtype=dtype)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return list(map(tuple, self.values[key]))
        else:
            return tuple(self.values[key])
        
    def __iter__(self):
        for v in self.values:
            yield tuple(v)
    
    def iter(self, levels=None):
        if levels is not None:
            if not isinstance(levels, list):
                levels = [levels]
            if all(isinstance(l, basestring) for l in levels):
                for i, l in enumerate(levels):
                    levels[i] = self.names.index(l)
            levels = [self.levels[l] for l in levels]
            values = list(itertools.product(*levels))
        else:
            values = self.values            
            
        for v in values:
            yield tuple(v)
    
    def iterargs(self):
        for v in self.values:
            yield dict(zip(self.names, v))
    
    def __len__(self):
        return len(self.levels)

#------------------------------------------------------------------------------
class SpectraResults(object): 
    
    def __init__(self, index, data):
        self.index = index
        self.data = data
 
    def from_files(cls, index, results_dir, basename):
        self.index = index
        self.data = {}
        for i, f in enum_files(results_dir, basename, index):
            self.data[i] = load_data(f)
        
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.slice(*key)
        else:
            return self.slice(key)
             
    def slice(self, *key):
        
        index = [None]*len(self.index.names)
        if not isinstance(key, (tuple, list)): key = (key, )
        for k in key:
            ikey = [i for i, level in enumerate(self.index.levels) if k in level]
            if len(ikey):
                index[ikey[0]] = k
        toret = {k:v for k,v in self.data.items() if all(i2 is None or i1 == i2 for (i1,i2) in zip(k,index))}
        if len(toret) == 1:
            return toret[toret.keys()[0]]
        else:
            return toret
        
    def __iter__(self):
        return self.index.__iter__()
    
    def iter(self, levels=None):
        return self.index.iter(levels)
            
    def enum(self):
        for i in self:
            yield i, self.data[i]
    
#------------------------------------------------------------------------------        
class HaloSpectraResults:
    
    def __init__(self, Phh, Phm, Pmm, bias_file=None):
        
        self.data = {}
        self.data['hh'] = Phh
        self.data['hm'] = Phm
        self.data['mm'] = Pmm
        self._add_errors()
    
        if bias_file is not None:
            self.biases = pickle.load(open(bias_file, 'r'))
            self._add_stoch()

    def __getitem__(self, key):
        if key in self.data.keys():
            return self.data[key]
        else:
            raise KeyError("key not understood in HaloSpectraResults")
            
    def common_iter(self):
        matching = set(self['hh'].index.names) & set(self['hm'].index.names) & set(self['mm'].index.names)
        for i in self['hh'].iter(levels=list(matching)):
            yield i
            
    def common_enum(self):
        matching = set(self['hh'].index.names) & set(self['hm'].index.names) & set(self['mm'].index.names)
        for i in self['hh'].iter(levels=list(matching)):
            yield i, (self['hh'][i], self['hm'][i], self['mm'][i])
        
    def _add_errors(self):
        
        # first do Phh and Pmm
        for key in ['hh', 'mm', 'hm']:
            
            # enumerate each spectra type
            for i, data in self[key].enum():
                d = data.data
                
                # compute the error
                with np.errstate(invalid='ignore'):
                    err = (2./d['modes'])**0.5 * d['power']
                dtype = d.dtype.descr
                if 'error' not in d.dtype.names:
                    dtype += [('error', 'f8')]
                new = np.zeros(d.shape, dtype=dtype)
                for name in d.dtype.names:
                    new[name] = d[name]
                new['error'] = err
                self.data[key][i].data = new

        # now do the cross
        for i, (Phhs, Phms, Pmm) in self.common_enum():
            for j in Phms:
                Phm, Phh = Phms[j], Phhs[j]
                with np.errstate(invalid='ignore'):
                    err = Phm['error'] / 2**0.5
                    err += (1./Phm['modes'])**0.5 * (Phh['power'] + Pmm['power'])
                self.data['hm'][j].data['error'] = err
                
        
    def _add_stoch(self):

        d = [{}, {}]
        for i, (Phhs, Phms, Pmm) in self.common_enum():
            for index in Phms:
                Phm, Phh = Phms[index], Phhs[index]
                b1 = 1. #self.biases[index]
                
                # output data array
                dtype = [('power', 'f8'), ('error', 'f8')]
                if hasattr(Phh, 'k_center'): dtype += [('k_center', 'f8')]
                if hasattr(Phh, 'mu_center'): dtype += [('k_center', 'f8')]
                data =  [np.empty(len(Phm['power']), dtype=dtype), 
                          np.empty(len(Phm['power']), dtype=dtype)]

                # type A stochasticity
                lam_a = Phh['power'] - 2*b1*Phm['power'] + b1**2*Pmm['power']
                err_a = (Phh['error']**2 + (2*b1*Phm['error'])**2 + (b1**2*Pmm['error'])**2)**0.5
                
                # type B stochasticity
                b1k = Phm['power'] / Pmm['power']
                b1k_err = b1k * ((Phm['error']/Phm['power'])**2 + (Pmm['error']/Pmm['power'])**2)**0.5
                lam_b = Phh['power'] - b1k**2 * Pmm['power']
                err_b = (Phh['error']**2 + (2*b1k*Pmm['power']*b1k_err)**2 + (b1k*Pmm['error'])**2)**0.5
        
                # save
                for i in range(2):
                    data[i]['power'] = lam_a
                    data[i]['error'] = err_a
                    if hasattr(Phh, 'k_center'): data[i]['k_center'] = Phh.k_center
                    if hasattr(Phh, 'mu_center'): data[i]['mu_center'] = Phh.mu_center
                
                    d[i][index] = data[i]
                
        self.data['stoch_a'] = SpectraResults(self['hh'].index, d[0])
        self.data['stoch_b'] = SpectraResults(self['hh'].index, d[1])
#------------------------------------------------------------------------------                  
                
            
        
        
                
                
                