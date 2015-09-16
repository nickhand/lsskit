class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
     
    def __getattribute__(self, key):
        try:
            return dict.__getattribute__(self, key)  
        except:
            return AttrDict()
        
from .theory.base import BaseTheoryParams
from .data import PkmuDataParams, PoleDataParams
from .driver import DriverParams