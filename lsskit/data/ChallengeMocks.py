from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet
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
    def get_Pgal(self, spacing="dk005", scaled=False, Nmu=5):
        """
        Return the total galaxy spectrum in redshift space
        
        Parameters
        ----------
        spacing : str, optional
            the tag used to identify results with different k spacing, i.e. `dk005`
        scaled : bool, optional
            return the results that have/haven't been scaled by AP factors
        Nmu : int, optional
            the number of mu bins
        """
        tag = 'unscaled' if not scaled else 'scaled'
        if spacing: spacing = '_' + spacing
        name = '_Pgal%s_%s_Nmu%d' %(spacing, tag, Nmu)
        try:
            return getattr(self, name)
        except AttributeError:
            
            # load the data from file
            basename = 'pkmu_challenge_box{box}_%s%s_Nmu%d.dat' %(tag, spacing, Nmu)
            coords = [self.boxes]
            d = os.path.join(self.root, 'pkmu')
            Pgal = self.reindex(SpectraSet.from_files(d, basename, coords, ['box']), self.dk)
            
            # add the errors
            Pgal.add_errors()
            setattr(self, name, Pgal)
            return Pgal
            
    #--------------------------------------------------------------------------
    # multipoles data
    #--------------------------------------------------------------------------
    def get_poles(self, spacing="dk005", scaled=False):
        """
        Return the total galaxy spectrum multipoles in redshift space
        """
        tag = 'unscaled' if not scaled else 'scaled'
        if spacing: spacing = '_'+spacing
        name = '_poles%s_%s' %(spacing, tag)
        try:
            return getattr(self, name)
        except AttributeError:
            
            # load the data from file
            basename = '{ell}_challenge_box{box}_%s%s.dat' %(tag, spacing)
            coords = [self.boxes, ['mono', 'quad', 'hexadec']]
            d = os.path.join(self.root, 'poles')
            P = self.reindex(SpectraSet.from_files(d, basename, coords, ['box', 'ell']), self.dk)
            
            P.coords['ell'] = [0, 2, 4]
            setattr(self, name, P)
            return P
    
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
                if box in ['A', 'B', 'F', 'G']:
                    stats[box]['qperp'] = 0.998753592
                    stats[box]['qpar'] = 0.9975277944
                    stats[box]['z'] = 0.562
                    stats[box]['box_size'] = 2500.
                elif box == 'C':
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