from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, HaloSpectraSet, utils
import os

class RunPBGalaxy(PowerSpectraLoader):
    name = "RunPBGalaxy"
    a = ['0.6452']
    spectra = ['gg', 'cc', 'cAcA', 'cAcB', 'cBcB', 'cs', 'cAs', 'cBs', 'ss', 'sAsA', 'sAsB', 'sBsB']
    samples = ['gal', 'cen', 'cenA', 'cenB', 'sat', 'satA', 'satB']
    
    def __init__(self, root, realization='10mean'):
        
        # store the root directory and the realization
        self.root = root
        self.tag = realization
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
        
    #--------------------------------------------------------------------------
    # galaxy data
    #--------------------------------------------------------------------------
    def get_Pgal(self, space='real'):
        """
        Return galaxy component spectra
        """
        try:
            return getattr(self, '_Pgal_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'galaxy', space)
            if space == 'real':
                basename = 'pk_{sample}_runPB_%s.dat' %self.tag
            else:
                basename = 'pkmu_{sample}_runPB_%s_Nmu5.dat' %self.tag
                
            coords = [self.a, self.spectra]
            Pgal = SpectraSet.from_files(d, basename, coords, ['a', 'sample'])
            
            # add the errors
            Pgal.add_errors()
            crosses = {'cAcB':['cAcA', 'cBcB'], 'cs':['cc', 'ss'], 'cAs':['cAcA', 'ss'], 'cBs':['cBcB', 'ss'], 'sAsB':['sAsA', 'sBsB']}
            for x in crosses:
                this_cross = Pgal.sel(a='0.6452', sample=x).values
                k1, k2 = crosses[x]
                utils.add_errors(this_cross, Pgal.sel(a='0.6452', sample=k1).values, Pgal.sel(a='0.6452', sample=k2).values)
                
            setattr(self, '_Pgal_'+space, Pgal)
            return Pgal
    
    def get_gal_x_matter(self, space='real'):
        """
        Return galaxy component spectra x matter
        """
        try:
            return getattr(self, '_Pgal_x_mm_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'galaxy', space)
            if space == 'real':
                basename = 'pk_{sample}_x_matter_runPB_%s.dat' %self.tag
            else:
                basename = 'pkmu_{sample}_x_matter_runPB_%s_Nmu5.dat' %self.tag
            coords = [['0.6452'], self.samples]
            Pgal = SpectraSet.from_files(d, basename, coords, ['a', 'sample'])
            
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