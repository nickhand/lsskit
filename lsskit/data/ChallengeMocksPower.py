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
            Pgal = self.reindex(Pgal, 'k', self.dk, weights='modes')
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
            poles = self.reindex(poles, 'k', self.dk, weights='modes')

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
    def get_Pgal(self, space='redshift', spacing="dk005", 
                    subtract_shot_noise=False, Nmu=100, scaled=False, average=None, tag=""):
        """
        Return the total galaxy spectrum in redshift space
        """
        if space not in ['redshift', 'real']:
            raise ValueError("`space` must be real' or 'redshift'")
            
        if average is not None:
            if isinstance(average, str):
                average = [average]
        else:
            average = []
                
        # determine the tag
        scaled_tag = 'scaled' if scaled else 'unscaled'
        _spacing = spacing
        if spacing: spacing = '_'+spacing            
        name = '_Pgal%s_%s_Nmu%d_%s' %(spacing, space, Nmu, scaled_tag)
        if len(average):
            name += '_' + '_'.join(average)        
        
        if tag: 
            tag = '_'+tag
            name += tag
            
        try:
            return getattr(self, name)
        except AttributeError:
            
            # load the data from file
            if space == 'redshift':
                basename = 'pkmu_challenge_boxN{box}_%s%s_Nmu%d_{los}los%s.dat' %(scaled_tag, spacing, Nmu, tag)
                coords = [self.los, self.boxes]
                dims = ['los', 'box']
            else:
                basename = 'pkmu_challenge_boxN{box}_real_%s%s_Nmu100%s.dat' %(scaled_tag, spacing, tag)
                coords = [self.boxes]
                dims = ['box']
            d = os.path.join(self.root, 'power')
            
            loader = io.load_power
            kwargs = {'sum_only':['modes'], 'force_index_match':True}
            Pgal = SpectraSet.from_files(loader, d, basename, coords, dims, args=('2d',), kwargs=kwargs)
            
            # take the real part
            for key in Pgal.ndindex():
                pkmu = (Pgal.loc[key]).get()
                pkmu['power'] = pkmu['power'].real
                
            if subtract_shot_noise:
                for key in Pgal.ndindex():
                    p = Pgal.loc[key].get()
                    p['power'] = p['power'] - p.attrs['volume'] / p.attrs['N1']
                        
            if len(average):
                Pgal = Pgal.average(axis=average, weights='modes')
            
            if isinstance(Pgal, SpectraSet):
                # reindex and add the errors
                Pgal = self.reindex(Pgal, 'k', self.dk, weights='modes')
                Pgal.add_power_errors()
            else:
                
                # reindex
                if self.dk is not None:
                    Pgal = Pgal.reindex('k', self.dk, weights='modes', force=True)

                # add errors
                Pgal['error'] =  (2./Pgal['modes'])**0.5 * Pgal['power']
            
            setattr(self, name, Pgal)
            return Pgal
            
    #--------------------------------------------------------------------------
    # multipoles data
    #--------------------------------------------------------------------------
    def get_poles(self, space='redshift', subtract_shot_noise=True, spacing="dk005", Nmu=100, 
                    scaled=False, average=None, tag="", include_tetrahex=False):
        """
        Return the N-series multipoles in redshift space
        """
        if space not in ['redshift', 'real']:
            raise ValueError("`space` must be real' or 'redshift'")
            
        if average is not None:
            if isinstance(average, str):
                average = [average]
        else:
            average = []
            
        # determine the tag
        scaled_tag = 'scaled' if scaled else 'unscaled'
        _spacing = spacing
        if spacing: spacing = '_'+spacing        
        name = '_nseries_poles_%s_%s%s' %(space, scaled_tag, spacing)
        if len(average):
            name += '_' + '_'.join(average)

        tag_ = '_'+tag if tag else ''
        if tag: name += tag_
        
        try:
            return getattr(self, name)
        except AttributeError:
            
            # load the data from file
            if space == 'redshift':
                basename = 'poles_challenge_boxN{box}_%s%s_Nmu%d_{los}los%s.dat' %(scaled_tag, spacing, Nmu, tag_)
                coords = [self.los, self.boxes]
                dims = ['los', 'box']
            else:
                basename = 'poles_challenge_boxN{box}_real_%s%s_Nmu100%s.dat' %(scaled_tag, spacing, tag_)
                coords = [self.boxes]
                dims = ['box']
            d = os.path.join(self.root, 'poles')
            
            loader = io.load_power
            mapcols = {'power_0.real':'mono', 'power_2.real':'quad', 'power_4.real':'hexadec'}
            usecols = ['k', 'mono', 'quad', 'hexadec', 'modes']
            if include_tetrahex: 
                mapcols['power_6.real'] = 'tetrahex'
                usecols.append('tetrahex')
            kwargs = {'usecols':usecols, 'mapcols':mapcols}
            poles = SpectraSet.from_files(loader, d, basename, coords, dims, args=('1d',), kwargs=kwargs)

            # reindex and add the errors
            poles = self.reindex(poles, 'k', self.dk, weights='modes')

            # now convert
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            if include_tetrahex: 
                ells.append(('tetrahex', 6))
            poles = tools.unstack_multipoles(poles, ells, 'power')

            if subtract_shot_noise:
                for key in poles.ndindex():
                    if key['ell'] == 0:
                        p = poles.loc[key].get()
                        p['power'] = p['power'] - p.attrs['volume'] / p.attrs['N1']

            # average?
            if len(average):
                poles = poles.average(axis=average, weights='modes')

            # add the errors
            pkmu = self.get_Pgal(space=space, spacing=_spacing, Nmu=Nmu, average=average, scaled=scaled, tag=tag)
            poles.add_power_pole_errors(pkmu)
            
            setattr(self, name, poles)
            return poles
            
    def get_shifted_poles(self, name, tag="", average=None, subtract_shot_noise=True):
        """
        Return the N-series multipoles in redshift space, which haven been shifted in
        space (and computed with Bianchi algorithm)
        """
        if name not in ['boss_like', 'plane_parallel']:
            raise ValueError("please choose a name from 'boss_like' or 'plane_parallel'")
            
        tag_ = '_'+tag if tag else ''
        if tag: name += tag_
        
        if average is not None:
            if isinstance(average, str):
                average = [average]
        else:
            average = []
            
        name_ = name
        if len(average):
            name_ += '_' + '_'.join(average)
        
        try:
            return getattr(self, 'shifted_poles_%s' %name_)
        except AttributeError:
            
            # load the data from file
            basename = 'poles_challenge_boxN{box}_shifted_unscaled_shift{shift}_%s.dat' %name
            coords = [self.boxes, [0, 1, 2]]
            d = os.path.join(self.root, 'poles')
            
            loader = io.load_power
            mapcols = {'power_0.real':'mono', 'power_2.real':'quad', 'power_4.real':'hexadec'}
            usecols = ['k', 'mono', 'quad', 'hexadec', 'modes']
            kwargs = {'usecols':usecols, 'mapcols':mapcols}
            poles = SpectraSet.from_files(loader, d, basename, coords, ['box', 'shift'], args=('1d',), kwargs=kwargs)

            # reindex and add the errors
            poles = self.reindex(poles, 'k', self.dk, weights='modes')

            # now convert
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            poles = tools.unstack_multipoles(poles, ells, 'power')

            if subtract_shot_noise:
                for key in poles.ndindex():
                    if key['ell'] == 0:
                        p = poles.loc[key].get()
                        p['power'] = p['power'] - p.attrs['shot_noise']

            # average?
            if len(average):
                poles = poles.average(axis=average, weights='modes')
            
            setattr(self, name, poles)
            return poles