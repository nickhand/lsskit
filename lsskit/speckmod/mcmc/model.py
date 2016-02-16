import numpy as np
from sklearn import preprocessing


class ScaledModel(object):
    """
    A model class that preprocesses the data, removing 
    the mean and normalizing to unit variance, for 
    both the independent and dependent variables
    """
    def __init__(self, x, y, yerr=None):
        """
        Parameters
        ----------
        X : array_like
            the independent variables
        Y : array_like
            the dependent variables
        """
        self.x = x
        self.y = y
        self.yerr = yerr
        
    def transform_x(self, d):
        """
        Transform the input array using the `x_scaler`
        """
        if d.ndim == 1: d = d.reshape(-1,1)
        return np.squeeze(self.x_scaler.transform(d))
        
    def transform_y(self, d):
        """
        Transform the input array using the `y_scaler`
        """
        return np.squeeze(self.y_scaler.transform(d.reshape(-1,1)))
    
    @property
    def x_scaler(self):
        """
        Preprocess scaler for the independent variables
        """
        try:
            return self._x_scaler
        except AttributeError:
            x = self.x
            if x.ndim == 1: x = x.reshape(-1,1)
            self._x_scaler = preprocessing.StandardScaler(copy=True).fit(x)
            return self._x_scaler
            
    @property
    def y_scaler(self):
        """
        Preprocess scaler for the dependent variable
        """
        try:
            return self._y_scaler
        except AttributeError:
            self._y_scaler = preprocessing.StandardScaler(copy=True).fit(self.y.reshape(-1,1))
            return self._y_scaler
        
    @property
    def x_scaled(self):
        """
        The scaled `x` variable
        """
        try:
            return self._x_scaled
        except AttributeError:
            self._x_scaled = self.transform_x(self.x)
            return self._x_scaled
            
    @property
    def y_scaled(self):
        """
        The scaled `y` variable
        """
        try:
            return self._y_scaled
        except AttributeError:
            self._y_scaled = self.transform_y(self.y)
            return self._y_scaled
            
    @property
    def yerr_scaled(self):
        """
        The scaled `yerr` variable
        """
        try:
            return self._yerr_scaled
        except AttributeError:
            if self.yerr is not None:
                self._yerr_scaled = self.yerr / self.y_scaler.scale_
                return self._yerr_scaled
            else:
                return None
        
        