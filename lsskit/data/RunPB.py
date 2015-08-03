from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, HaloSpectraSet, utils
import os
import pickle

class RunPB(PowerSpectraLoader):
    name = "RunPB"
    a = ['0.5714', '0.6061', '0.6452', '0.6667', '0.6897', '0.7143', '0.8000', '0.9091', '1.0000']
    mass = range(8)
    
    def __init__(self, root, realization='10mean'):
        
        # store the root directory and the realization
        self.root = root
        self.tag = realization
      
    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)  
        
    #--------------------------------------------------------------------------
    # data accessors
    #--------------------------------------------------------------------------
    def get_Phh(self, space='real'):
        """
        Return Phh in the space specified, either `real` or `redshift`
        """
        try:
            return getattr(self, '_Phh_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo', space)
            if space == 'real':
                basename = 'pk_hh{mass}_runPB_%s_{a}.dat' %self.tag
            else:
                basename = 'pkmu_hh{mass}_runPB_%s_{a}_Nmu5.dat' %self.tag
                
            coords = [self.a, self.mass]
            Phh = SpectraSet.from_files(d, basename, coords, ['a', 'mass'])
            Phh.add_errors()
            setattr(self, '_Phh_'+space, Phh)
            return Phh
            
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
            Pmm = SpectraSet.from_files(d, basename, coords, ['a'])
            Pmm.add_errors()
            setattr(self, '_Pmm_'+space, Pmm)
            return Pmm
            
    def get_Phm(self, space='real'):
        """
        Return Phm in the space specified, either `real` or `redshift`
        """
        try:
            return getattr(self, '_Phm_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo-matter', space)
            if space == 'real':
                basename = 'pk_hm{mass}_runPB_%s_{a}.dat' %self.tag
            else:
                basename = 'pkmu_hm{mass}_runPB_%s_{a}_Nmu5.dat' %self.tag
                
            coords = [self.a, self.mass]
            Phm = SpectraSet.from_files(d, basename, coords, ['a', 'mass'])
            Phm.add_errors(self.get_Phh(space), self.get_Pmm(space))
            setattr(self, '_Phm_'+space, Phm)
            return Phm
            
    def get_lambda(self, kind='A', space='real', bias_file=None):
        """
        Return the stochasticity 
        """
        name = '_lambda%s_%s' %(kind, space)
        try:
            return getattr(self, name)
        except AttributeError:

            biases = self.get_halo_biases(bias_file)
            data = HaloSpectraSet(self.get_Phh(space), self.get_Phm(space), self.get_Pmm(space), biases)
            lam = data.to_lambda(kind)
            setattr(self, name, lam)
            return lam
            

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
                
            coords = [['gg', 'cc', 'cAcA', 'cAcB', 'cBcB', 'cs', 'cAs', 'cBs', 'ss', 'sAsA', 'sAsB', 'sBsB']]
            Pgal = SpectraSet.from_files(d, basename, coords, ['sample'])
            
            # add the errors
            Pgal.add_errors()
            crosses = {'cAcB':['cAcA', 'cBcB'], 'cs':['cc', 'ss'], 'cAs':['cAcA', 'ss'], 'cBs':['cBcB', 'ss'], 'sAsB':['sAsA', 'sBsB']}
            for x in crosses:
                this_cross = Pgal.sel(sample=x).values
                k1, k2 = crosses[x]
                utils.add_errors(this_cross, Pgal.sel(sample=k1).values, Pgal.sel(sample=k2).values)
                
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
            names = ['gal', 'cen', 'cenA', 'cenB', 'sat', 'satA', 'satB']
            coords = [names]
            Pgal = SpectraSet.from_files(d, basename, coords, ['sample'])
            
            # now add errors, using Pmm at z = 0.55 and each galaxy auto spectrum
            Pgal_autos = self.get_Pgal(space=space)
            Pmm = self.get_Pmm(space=space).sel(a='0.6452').values
            auto_names = ['gg', 'cc', 'cAcA', 'cBcB', 'ss', 'sAsA', 'sBsB']
            keys = dict(zip(names, auto_names))
            for k in names:
                this_cross = Pgal.sel(sample=k).values
                Pgal_auto = Pgal_autos.sel(sample=keys[k]).values
                utils.add_errors(this_cross, Pgal_auto, Pmm)
            
            # set and return
            setattr(self, '_Pgal_x_mm_'+space, Pgal)
            return Pgal
    
    def get_gal_biases(self, filename=None):
        """
        Return the linear biases of each galaxy sample
        """
        try:
            return self._gal_biases
        except:
            if filename is None:
                filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/biases_galaxy_samples.pickle")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            biases = utils.load_data_from_file(filename, ['sample'], (7,))
            setattr(self, '_gal_biases', biases)
            return biases
    
    def get_gal_stats(self, filename=None):
        """
        Return a dictionary holding the galaxy sample statistics, fractions, etc
        """
        try:
            return self._gal_stats
        except:
            if filename is None:
                filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/galaxy_sample_stats.pickle")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            stats = pickle.load(open(filename, 'r'))
            setattr(self, '_gal_stats', stats)
            return stats
    
    def get_halo_biases(self, filename=None):
        """
        Return the linear biases of each halo mass bin 
        """
        try:
            return self._halo_biases
        except:
            if filename is None:
                filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/biases_halo_mass_bins.pickle")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            biases = utils.load_data_from_file(filename, ['a', 'mass'], (len(self.a), len(self.mass)))
            setattr(self, '_halo_biases', biases)
            return biases
            
    def get_halo_masses(self, filename=None):
        """
        Return the average mass of each halo mass bin 
        """
        try:
            return self._halo_masses
        except:
            if filename is None:
                filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/avg_halo_masses.pickle")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            masses = utils.load_data_from_file(filename, ['a', 'mass'], (len(self.a), len(self.mass)))
            setattr(self, '_halo_masses', masses)
            return masses
