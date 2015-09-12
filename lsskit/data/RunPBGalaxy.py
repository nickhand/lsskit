from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, utils, tools
import os

class RunPBGalaxy(PowerSpectraLoader):
    name = "RunPBGalaxy"
    a = ['0.6452']
    spectra = ['gg', 'cc', 'cAcA', 'cAcB', 'cBcB', 'cs', 'cAs', 'cBs', 'ss', 'sAsA', 'sAsB', 'sBsB']
    samples = ['gal', 'cen', 'cenA', 'cenB', 'sat', 'satA', 'satB']
    
    def __init__(self, root, realization='10mean', dk=None):
        
        # store the root directory and the realization
        self.root = root
        self.tag = realization
        self.dk = dk
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls) 
    
    def get_Pmm(self, space='real'):
        """
        Return Pmm in the space specified, either `real` or `redshift`
        """
        try:
            return getattr(self, '_Pmm_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'matter', space)
            if space == 'real':
                basename = 'pk_mm_runPB_%s_{a}.dat' %self.tag
            else:
                basename = 'pkmu_mm_runPB_%s_{a}_Nmu5.dat' %self.tag
                
            coords = [self.a]
            Pmm = self.reindex(SpectraSet.from_files(d, basename, coords, ['a']), self.dk)
            Pmm.add_errors()
            setattr(self, '_Pmm_'+space, Pmm)
            return Pmm 
        
    #--------------------------------------------------------------------------
    # galaxy data
    #--------------------------------------------------------------------------
    def get_Pgal(self, space='real', spacing="", Nmu=5):
        """
        Return galaxy component spectra
        """
        if spacing: spacing = "_"+spacing
        if space == 'real':
            name = '_Pgal_%s%s' %(self.tag, spacing)
        else:
            name = '_Pgal_%s%s_Nmu%d' %(self.tag, spacing, Nmu)
        try:
            return getattr(self, name)
        except AttributeError:
            
            d = os.path.join(self.root, 'galaxy', space, 'power')
            if space == 'real':
                basename = 'pk_{sample}_runPB_%s%s.dat' %(self.tag, spacing)
            else:
                basename = 'pkmu_{sample}_runPB_%s%s_Nmu%d.dat' %(self.tag, spacing, Nmu)
                
            # load the data from file
            coords = [self.a, self.spectra]
            Pgal = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'sample'], ignore_missing=True), self.dk)
            
            # add the errors
            Pgal.add_errors()
            crosses = {'cAcB':['cAcA', 'cBcB'], 'cs':['cc', 'ss'], 'cAs':['cAcA', 'ss'], 'cBs':['cBcB', 'ss'], 'sAsB':['sAsA', 'sBsB']}
            for x in crosses:
                
                this_cross = Pgal.sel(a='0.6452', sample=x)
                if this_cross.isnull(): continue
                this_cross = this_cross.values
                k1, k2 = crosses[x]
                utils.add_errors(this_cross, Pgal.sel(a='0.6452', sample=k1).values, Pgal.sel(a='0.6452', sample=k2).values)
                
            if not Pgal.notnull().sum():
                raise ValueError("there appears to be no non-null entries -- something probably went wrong")
            setattr(self, name, Pgal)
            return Pgal
            
    def get_poles(self, space='redshift', spacing="dk005", Nmu=100):
        """
        Return galaxy component spectra multipoles
        """
        _spacing = spacing
        if spacing: spacing = "_"+spacing
        name = '_poles_%s%s_%s' %(self.tag, spacing, space)
        try:
            return getattr(self, name)
        except AttributeError:
            
            d = os.path.join(self.root, 'galaxy', space, 'poles')
            basename = 'poles_{sample}_runPB_%s%s_Nmu%d.dat' %(self.tag, spacing, Nmu)
                
            # load the data from file
            coords = [self.a, self.spectra]
            columns = ['k', 'mono', 'quad', 'hexadec', 'modes']
            poles = SpectraSet.from_files(d, basename, coords, ['a', 'sample'], ignore_missing=True, columns=columns)
            poles = self.reindex(poles, self.dk)
            if not poles.notnull().sum():
                raise ValueError("there appears to be no non-null entries -- something probably went wrong")
            
            # now convert
            pkmu = self.get_Pgal(space=space, spacing=_spacing, Nmu=Nmu)    
            ells = {'mono':0, 'quad':2, 'hexadec':4}
            toret = tools.format_multipoles_set(poles, pkmu, ells)
            
            setattr(self, name, toret)
            return toret
    
    def get_gal_x_matter(self, space='real'):
        """
        Return galaxy component spectra x matter
        """
        try:
            return getattr(self, '_Pgal_x_mm_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'galaxy', space, 'power')
            if space == 'real':
                basename = 'pk_{sample}_x_matter_runPB_%s.dat' %self.tag
            else:
                basename = 'pkmu_{sample}_x_matter_runPB_%s_Nmu5.dat' %self.tag
            coords = [['0.6452'], self.samples]
            Pgal = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'sample']), self.dk)
            
            # now add errors, using Pmm at z = 0.55 and each galaxy auto spectrum
            Pgal_autos = self.get_Pgal(space=space)
            Pmm = self.get_Pmm(space=space).sel(a='0.6452').values
            auto_names = ['gg', 'cc', 'cAcA', 'cBcB', 'ss', 'sAsA', 'sBsB']
            keys = dict(zip(self.samples, auto_names))
            for k in self.samples:
                this_cross = Pgal.sel(a='0.6452', sample=k).values
                Pgal_auto = Pgal_autos.sel(a='0.6452', sample=keys[k]).values
                utils.add_errors(this_cross, Pgal_auto, Pmm)
            
            # set and return
            setattr(self, '_Pgal_x_mm_'+space, Pgal)
            return Pgal
    
    #--------------------------------------------------------------------------
    # auxiliary data
    #--------------------------------------------------------------------------
    def get_gal_biases(self, filename=None):
        """
        Return the linear biases of each galaxy sample
        """
        try:
            return self._gal_biases
        except:
            import xray
            if filename is None:
                filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/biases_galaxy_samples.pickle")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            biases = utils.load_data_from_file(filename, ['a', 'sample'], (len(self.a), len(self.samples)))
            setattr(self, '_gal_biases', biases)
            return biases
    
    def get_gal_stats(self, filename=None):
        """
        Return a dictionary holding the galaxy sample statistics, fractions, etc
        """
        try:
            return self._gal_stats
        except:
            import pickle
            if filename is None:
                filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/galaxy_sample_stats.pickle")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            stats = pickle.load(open(filename, 'r'))
            setattr(self, '_gal_stats', stats)
            return stats