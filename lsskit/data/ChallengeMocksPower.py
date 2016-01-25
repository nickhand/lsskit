from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, tools, io
import os

class ChallengeMocksPower(PowerSpectraLoader):
    name = "ChallengeMocksPower"
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
            d = os.path.join(self.root, 'power')
            
            loader = io.load_power
            kwargs = {'sum_only':['modes'], 'force_index_match':True}
            Pgal = SpectraSet.from_files(loader, d, basename, coords, ['box'], args=('2d',), kwargs=kwargs)
            
            # reindex and add the errors
            Pgal = self.reindex(Pgal, 'k_cen', self.dk, weights='modes')
            Pgal.add_power_errors()
            
            setattr(self, name, Pgal)
            return Pgal
            
    #--------------------------------------------------------------------------
    # multipoles data
    #--------------------------------------------------------------------------
    def get_poles(self, spacing="dk005", scaled=False, Nmu=100):
        """
        Return the total galaxy spectrum multipoles in redshift space
        """
        _spacing = spacing
        tag = 'unscaled' if not scaled else 'scaled'
        if spacing: spacing = '_'+spacing
        name = '_poles%s_%s' %(spacing, tag)
        try:
            return getattr(self, name)
        except AttributeError:
            
            # load the data from file
            basename = 'poles_challenge_box{box}_%s%s_Nmu%d.dat' %(tag, spacing, Nmu)
            coords = [self.boxes]
            d = os.path.join(self.root, 'poles')
            
            loader = io.load_power
            columns = ['k', 'mono', 'quad', 'hexadec', 'modes']
            kwargs = {'sum_only':['modes'], 'force_index_match':True, 'columns':columns}
            poles = SpectraSet.from_files(loader, d, basename, coords, ['box'], args=('1d',), kwargs=kwargs)

            # reindex and add the errors
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')

            # now convert
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            toret = tools.unstack_multipoles(poles, ells, 'power')
            
            # add the errors
            pkmu = self.get_Pgal(scaled=scaled, spacing=_spacing, Nmu=Nmu)  
            toret.add_power_pole_errors(pkmu)
            
            setattr(self, name, toret)
            return toret
            
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
            
            
class NSeriesChallengeMocksPower(PowerSpectraLoader):
    name = "NSeriesChallengeMocksPower"
    boxes = range(1, 8)
    los = ['x', 'y', 'z']
    
    def __init__(self, root, dk=None):
        self.root = root
        self.dk = dk
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
        
    #--------------------------------------------------------------------------
    # power data
    #--------------------------------------------------------------------------            
    def get_Pgal(self, spacing="dk005", Nmu=100, average=None):
        """
        Return the total galaxy spectrum in redshift space
        """
        if average is not None:
            if isinstance(average, str):
                average = [average]
        else:
            average = []
                
        # determine the tag
        _spacing = spacing
        if spacing: spacing = '_'+spacing            
        name = '_Pgal%s_Nmu%d' %(spacing, Nmu)
        if len(average):
            name += '_' + '_'.join(average)        
        
        try:
            return getattr(self, name)
        except AttributeError:
            
            # load the data from file
            basename = 'pkmu_challenge_boxN{box}_scaled%s_Nmu%d_{los}los.dat' %(spacing, Nmu)
            coords = [self.los, self.boxes]
            d = os.path.join(self.root, 'power')
            
            loader = io.load_power
            kwargs = {'sum_only':['modes'], 'force_index_match':True}
            Pgal = SpectraSet.from_files(loader, d, basename, coords, ['los', 'box'], args=('2d',), kwargs=kwargs)
            
            # take the real part
            for key in Pgal.ndindex():
                pkmu = (Pgal.loc[key]).values
                pkmu['power'] = pkmu['power'].real
                
            if len(average):
                Pgal = Pgal.average(axis=average, weights='modes')
            
            if isinstance(Pgal, SpectraSet):
                # reindex and add the errors
                Pgal = self.reindex(Pgal, 'k_cen', self.dk, weights='modes')
                Pgal.add_power_errors()
            else:
                
                # reindex
                if self.dk is not None:
                    Pgal = Pgal.reindex('k_cen', self.dk, weights='modes', force=True)

                # add errors
                Pgal['error'] =  (2./Pgal['modes'])**0.5 * Pgal['power']
            
            setattr(self, name, Pgal)
            return Pgal
            
    #--------------------------------------------------------------------------
    # multipoles data
    #--------------------------------------------------------------------------
    def get_poles(self, spacing="dk005", Nmu=100, average=None):
        """
        Return the N-series multipoles in redshift space
        """
        if average is not None:
            if isinstance(average, str):
                average = [average]
        else:
            average = []
            
        # determine the tag
        _spacing = spacing
        if spacing: spacing = '_'+spacing        
        name = '_nseries_poles%s' %(spacing)
        if len(average):
            name += '_' + '_'.join(average)
        
        try:
            return getattr(self, name)
        except AttributeError:
            
            # load the data from file
            basename = 'poles_challenge_boxN{box}_scaled%s_Nmu%d_{los}los.dat' %(spacing, Nmu)
            coords = [self.los, self.boxes]
            d = os.path.join(self.root, 'poles')
            
            loader = io.load_power
            mapcols = {'power_0.real':'mono', 'power_2.real':'quad', 'power_4.real':'hexadec'}
            usecols = ['k', 'mono', 'quad', 'hexadec', 'modes']
            kwargs = {'sum_only':['modes'], 'force_index_match':True, 'usecols':usecols, 'mapcols':mapcols}
            poles = SpectraSet.from_files(loader, d, basename, coords, ['los', 'box'], args=('1d',), kwargs=kwargs)

            # reindex and add the errors
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')

            # now convert
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            toret = tools.unstack_multipoles(poles, ells, 'power')
            
            # average?
            if len(average):
                toret = toret.average(axis=average, weights='modes')
            
            # add the errors
            pkmu = self.get_Pgal(spacing=_spacing, Nmu=Nmu, average=average)  
            toret.add_power_pole_errors(pkmu)
            
            setattr(self, name, toret)
            return toret