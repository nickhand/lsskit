from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, utils
import os

class ChallengeMocks(PowerSpectraLoader):
    name = "ChallengeMocks"
    boxes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    def __init__(self, root, dk=None):
        self.root = root
        self.dk = dk
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
        
    #--------------------------------------------------------------------------
    # galaxy data
    #--------------------------------------------------------------------------
    def get_Pgal(self, scaled=False):
        """
        Return the total galaxy spectrum in redshift space
        """
        tag = 'unscaled' if not scaled else 'scaled'
        try:
            return getattr(self, '_Pgal_'+tag)
        except AttributeError:
            basename = 'pkmu_challenge_box{box}_scaled_Nmu5.dat'
            coords = [self.boxes]
            Pgal = self.reindex(SpectraSet.from_files(self.root, basename, coords, ['box']), self.dk)
            
            # add the errors
            Pgal.add_errors()
            setattr(self, '_Pgal_'+tag, Pgal)
            return Pgal
    

    def get_box_stats(self):
        """
        Return a dictionary holding the box size, redshift, scalings for each box
        """
        try:
            return self._gal_stats
        except:
            stats = {}
            for box in self.boxes:
                stats[box] = {}
                if box in ['A', 'F', 'G']:
                    stats[box]['qperp'] = 0.998753592
                    stats[box]['qpar'] = 0.9975277944
                    stats[box]['z'] = 0.562
                    stats[box]['box_size'] = 2500.
                elif box in ['B', 'C']:
                    stats[box]['qperp'] = 0.9875682111
                    stats[box]['qpar'] = 0.9751013789
                    stats[box]['z'] = 0.441
                    stats[box]['box_size'] = 2500.
                else:
                    stats[box]['qperp'] = 0.9916978595
                    stats[box]['qpar'] = 0.9834483344
                    stats[box]['z'] = 0.5
                    stats[box]['box_size'] = 2600.
                    
            setattr(self, '_box_stats', stats)
            return stats