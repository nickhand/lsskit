from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, HaloSpectraSet
import os

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
                    
            biases = HaloSpectraSet.load_biases(filename, ['a', 'mass'], (len(self.a), len(self.mass)))
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
                    
            masses = HaloSpectraSet.load_masses(filename, ['a', 'mass'], (len(self.a), len(self.mass)))
            setattr(self, '_halo_masses', masses)
            return masses
