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
    
    def __init__(self, root, dk=None):
        self.root = root
        self.dk = dk
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
        
    #--------------------------------------------------------------------------
    # power data
    #--------------------------------------------------------------------------
    def get_mean_Pgal(self, spacing="dk005", Nmu=100):
        """
        Return the mean galaxy spectrum in redshift space, averaged of `los` and box
        """
        if spacing: spacing += "_"
        name = '_mean_Pgal_%sNmu%d' %(spacing, Nmu)
        try:
            return getattr(self, name)
        except AttributeError:
            
            # form the filename
            basename = 'pkmu_challenge_nseries_scaled_%sNmu%d_mean.dat' %(spacing, Nmu)
            filename = os.path.join(self.root, 'power', basename)
            
            # load the data and possibly re-index
            kw = {'sum_only':['modes'], 'force_index_match':True}
            Pgal = io.load_power(filename, '2d', **kw)
            
            # reindex
            if self.dk is not None:
                Pgal = Pgal.reindex('k_cen', self.dk, weights='modes', force=True)

            # add errors
            Pgal['error'] =  (2./Pgal['modes'])**0.5 * Pgal['power']
            
            setattr(self, name, Pgal)
            return Pgal
            
    def get_Pgal(self, spacing="dk005", Nmu=100, los=""):
        """
        Return the total galaxy spectrum in redshift space
        """
        # determine the tag
        _spacing = spacing
        if spacing: spacing = '_'+spacing
        if los:
            tag = los+"los"
        else:
            tag = "mean"
            
        name = '_Pgal%s_%s_Nmu%d' %(spacing, tag, Nmu)
        try:
            return getattr(self, name)
        except AttributeError:
            
            # load the data from file
            basename = 'pkmu_challenge_boxN{box}_scaled%s_Nmu%d_%s.dat' %(spacing, Nmu, tag)
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
    def get_poles(self, spacing="dk005", los="", Nmu=100):
        """
        Return the N-series multipoles in redshift space
        """
        # determine the tag
        _spacing = spacing
        if spacing: spacing = '_'+spacing
        if los:
            tag = los+"los"
        else:
            tag = "mean"
        
        name = '_nseries_poles%s_%s' %(spacing, tag)
        try:
            return getattr(self, name)
        except AttributeError:
            
            # load the data from file
            basename = 'poles_challenge_boxN{box}_scaled%s_Nmu%d_%s.dat' %(spacing, Nmu, tag)
            coords = [range(1, 8)]
            d = os.path.join(self.root, 'poles')
            
            loader = io.load_power
            mapcols = {'power_0.real':'mono', 'power_2.real':'quad', 'power_4.real':'hexadec'}
            usecols = ['k', 'mono', 'quad', 'hexadec', 'modes']
            kwargs = {'sum_only':['modes'], 'force_index_match':True, 'usecols':usecols, 'mapcols':mapcols}
            poles = SpectraSet.from_files(loader, d, basename, coords, ['box'], args=('1d',), kwargs=kwargs)

            # reindex and add the errors
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')

            # now convert
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            toret = tools.unstack_multipoles(poles, ells, 'power')
            
            # add the errors
            pkmu = self.get_Pgal(los=los, spacing=_spacing, Nmu=Nmu)  
            toret.add_power_pole_errors(pkmu)
            
            setattr(self, name, toret)
            return toret
            
    def get_mean_poles(self, spacing="dk005", Nmu=100):
        """
        Return the mean galaxy spectrum multipoles, averaged over los and box
        """
        _spacing = spacing
        if spacing: spacing = '_'+spacing
        name = '_mean_poles%s' %(spacing)
        
        try:
            return getattr(self, name)
        except AttributeError:
            data = []

            basename = 'poles_challenge_nseries_scaled%s_Nmu%d_mean.dat' %(spacing, Nmu)
            filename = os.path.join(self.root, 'poles', basename)
            
            mapcols = {'power_0.real':'mono', 'power_2.real':'quad', 'power_4.real':'hexadec'}
            usecols = ['k', 'mono', 'quad', 'hexadec', 'modes']
            kw = {'force_index_match':True, 'sum_only':['modes'], 'usecols':usecols, 'mapcols':mapcols}
            poles = io.load_power(filename, '1d', **kw)
            
            # reindex
            if self.dk is not None:
                poles = poles.reindex(self.dk, weights='modes', force=True)
            
            # now convert
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            poles = tools.unstack_multipoles_one(poles, ells, 'power')
            
            # make the SpectraSet
            poles = SpectraSet(poles, coords=[[0, 2, 4]], dims=['ell'])
            
            # add the errors
            pkmu = self.get_mean_Pgal(spacing=_spacing, Nmu=Nmu) 
            poles.add_power_pole_errors(pkmu)

            setattr(self, name, poles)
            return poles
    
