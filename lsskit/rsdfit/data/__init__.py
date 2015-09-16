from .. import AttrDict
import numpy as np

class DataParams(object):
    """
    Class to hold and manipulate the data parameters required for fitting
    """
    statistics = None    
    valid_params = ['statistics', 'fitting_range', 'data_file', 'usedata', 'columns',
                    'covariance', 'covariance_rescaling', 'covariance_Nmocks', 'rescale_inverse_covariance']
    
    def __iter__(self):
        """
        Iterate over the valid parameters
        """
        for name in self.valid_params:
            yield name
    
    def __str__(self):
        toret = []
        for k in self:
            v = getattr(self, k, None)
            toret.append("data.%s = %s" %(k,repr(v)))
        return "\n".join(toret)
        
    def update(self, filename, ignore=[]):
        """
        Update the parameters from file with the synax `data.param_name`
        """
        ns = {'data':self}
        for name in ignore:
            ns[name] = AttrDict()
        execfile(filename, ns)
        
    @classmethod
    def from_file(cls, filename, **kwargs):
        """
        Return an instance of the class, adfter updating the parameters
        from the specified file
        """
        toret = cls()
        toret.update(filename, **kwargs)
        return toret
        
    def to_file(self, filename, mode='w'):
        """
        Write out all valid parameters to the specified file
        """
        with open(filename, mode=mode) as ff:
            ff.write("#{0}\n# data params\n#{0}\n".format("-"*78))
            for name in self:
                if not hasattr(self, name):
                    raise ValueError('please specify `%s` attribute before writing to file' %name)
                par = getattr(self, name)
                ff.write("data.%s = %s\n" %(name, repr(par)))
                        
    @property
    def size(self):
        """
        The number of data measurements
        """
        return len(self.usedata)
        

    @property
    def rescale_inverse_covariance(self):
        """
        Whether or not we want to rescale the inverse covariance matrix
        """
        try:
            return self._rescale_inverse_covariance
        except:
            return False
            
    @property
    def covariance_Nmocks(self):
        """
        the number of mocks to use when rescaling the inverse covariance matrix
        """
        try:
            return self._covariance_Nmocks
        except:
            if self.rescale_inverse_covariance:
                raise AttributeError("please specify `covariance_Nmocks` since `rescale_inverse_covariance` is `True`")
            return 0
            
    @property
    def covariance_rescaling(self):
        """
        Rescaling factor for covariance matrix
        """
        try:
            return self._covariance_rescaling
        except:
            return 1.0
    
    @property
    def columns(self):
        """
        Column numbers corresponding to the measurements in the data file
        """
        return getattr(self, '_columns', range(len(self.statistics)))
    
    @columns.setter
    def columns(self, val):
        self._columns = val

    @property
    def usedata(self):
        """
        Slice the statistics using these indices
        """
        return getattr(self, '_usedata', range(len(self.statistics)))
    
    @usedata.setter
    def usedata(self, val):
        self._usedata = val
        
    @property
    def kmin(self):
        try:
            return self._kmin
        except AttributeError:
            raise AttributeError("please set `kmin`")
    
    @kmin.setter
    def kmin(self, val):
        self._kmin = np.empty(self.size)
        try:
            self._kmin[:] = val
        except:
            raise ValueError("`kmin` should be a scalar or list of length %d" %self.size)
            
    @property
    def kmax(self):
        try:
            return self._kmax
        except AttributeError:
            raise AttributeError("please set `kmax`")
    
    @kmax.setter
    def kmax(self, val):
        self._kmax = np.empty(self.size)
        try:
            self._kmax[:] = val
        except:
            raise ValueError("`kmax` should be a scalar or list of length %d" %self.size)
        
    @property
    def fitting_range(self):
        return zip(self.kmin, self.kmax)
        
        
class PkmuDataParams(DataParams):
    """
    Data params for P(k,mu) measurement with 5 mu bins
    """
    statistics = ['pkmu_0.1', 'pkmu_0.3', 'pkmu_0.5', 'pkmu_0.7', 'pkmu_0.9']
    
class PoleDataParams(DataParams):
    """
    Data params for multipoles measurements of ell = 0, 2, 4
    """
    statistics = ['ell_0', 'ell_2', 'ell_4']
    
            
    
    
        