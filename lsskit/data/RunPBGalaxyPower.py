from lsskit import numpy as np
from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, utils, tools, io
import os

class RunPBGalaxy(PowerSpectraLoader):
    
    name = "RunPBGalaxyPower"
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
        Return density auto-correlation in the space specified, 
        either `real` or `redshift`
        """
        name = '_Pmm_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            tag = self.tag
            columns = None
            d = os.path.join(self.root, 'matter', space, 'power')
            if space == 'real':
                basename = 'pk_mm_runPB_%s_{a}.dat' %tag
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_mm_runPB_%s_{a}_Nmu5.dat' %tag
                
            coords = [self.a]
            loader = io.load_power
            kwargs = {}
            if columns is not None: kwargs['columns'] = columns
            Pmm = SpectraSet.from_files(loader, d, basename, coords, dims=['a'], 
                                            args=('1d',), kwargs=kwargs)
            
            # reindex
            Pmm = self.reindex(Pmm, 'k_cen', self.dk, weights='modes')
            
            # add errors
            Pmm.add_power_errors()
            
            setattr(self, name, Pmm)
            return Pmm
            
    #--------------------------------------------------------------------------
    # galaxy data
    #--------------------------------------------------------------------------
    def get_Pgal(self, space='real', spacing="", Nmu=5, collisions=False):
        """
        Return galaxy component spectra
        """
        # format the name options
        if spacing: spacing = "_"+spacing
        colls_tag = "" if not collisions else "_collisions"
        if space == 'real':
            name = '_Pgal_%s%s%s' %(self.tag, spacing, colls_tag)
        else:
            name = '_Pgal_%s%s_Nmu%d%s' %(self.tag, spacing, Nmu, colls_tag)
        
        try:
            return getattr(self, name)
        except AttributeError:
            
            columns = None
            d = os.path.join(self.root, 'galaxy', space, 'power')
            if space == 'real':
                basename = 'pk_{sample}%s_runPB_%s%s.dat' %(colls_tag, self.tag, spacing)
                mode = '1d'
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_{sample}%s_runPB_%s%s_Nmu%d.dat' %(colls_tag, self.tag, spacing, Nmu)
                mode = '2d'
                
            # load the data from file
            coords = [self.a, self.spectra]
            dims = ['a', 'sample']
            
            loader = io.load_power
            kw = {'force_index_match':True, 'sum_only':['modes']}
            if columns is not None: kw['columns'] = columns
            Pgal = SpectraSet.from_files(loader, d, basename, coords, dims, ignore_missing=True, args=(mode,), kwargs=kw)
            
            # reindex
            Pgal = self.reindex(Pgal, 'k_cen', self.dk, weights='modes')
            
            # add the errors
            Pgal.add_power_errors()
            crosses = {'cAcB':['cAcA', 'cBcB'], 'cs':['cc', 'ss'], 'cAs':['cAcA', 'ss'], 
                        'cBs':['cBcB', 'ss'], 'sAsB':['sAsA', 'sBsB']}
                        
            for x in crosses:
                this_cross = Pgal.sel(a='0.6452', sample=x)
                if this_cross.isnull(): continue
                this_cross = this_cross.values
                k1, k2 = crosses[x]
                
                a = Pgal.sel(a='0.6452', sample=k1).values
                b = Pgal.sel(a='0.6452', sample=k2).values 
                utils.add_power_errors(this_cross, a, b)
                
            if not Pgal.notnull().sum():
                raise ValueError("there appears to be no non-null entries -- something probably went wrong")
            setattr(self, name, Pgal)
            return Pgal
            
    def get_poles(self, space='redshift', spacing="dk005", Nmu=100, collisions=False):
        """
        Return galaxy component spectra multipoles
        """
        _spacing = spacing
        if spacing: spacing = "_"+spacing
        colls_tag = "" if not collisions else "_collisions"
        name = '_poles_%s%s_%s%s' %(self.tag, spacing, space, colls_tag)
        
        try:
            return getattr(self, name)
        except AttributeError:
            
            d = os.path.join(self.root, 'galaxy', space, 'poles')
            basename = 'poles_{sample}%s_runPB_%s%s_Nmu%d.dat' %(colls_tag, self.tag, spacing, Nmu)
                
            # load the data from file
            coords = [self.a, self.spectra]
            dims = ['a', 'sample']
            
            loader = io.load_power
            columns = ['k', 'mono', 'quad', 'hexadec', 'modes']
            kw = {'columns':columns, 'force_index_match':True, 'sum_only':['modes']}
            
            # load and reindex
            poles = SpectraSet.from_files(loader, d, basename, coords, dims, ignore_missing=True, args=('1d',), kwargs=kw)
            poles = self.reindex(poles, 'k_cen', self.dk, weights='modes')
            if not poles.notnull().sum():
                raise ValueError("there appears to be no non-null entries -- something probably went wrong")
            
            # unstck multipoles
            ells = [('mono',0), ('quad', 2), ('hexadec', 4)]
            toret = tools.unstack_multipoles(poles, ells, 'power')
            
            # add errors
            pkmu = self.get_Pgal(space=space, spacing=_spacing, Nmu=Nmu)
            toret.add_power_pole_errors(pkmu)
            
            setattr(self, name, toret)
            return toret
    
    def get_gal_x_matter(self, space='real'):
        """
        Return galaxy component spectra x matter
        """
        name = '_Pgal_x_mm_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            d = os.path.join(self.root, 'galaxy', space, 'power')
            if space == 'real':
                basename = 'pk_{sample}_x_matter_runPB_%s.dat' %self.tag
                mode = '1d'
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_{sample}_x_matter_runPB_%s_Nmu5.dat' %self.tag
                
                
            coords = [['0.6452'], self.samples]
            dims = ['a', 'sample']
            
            # load
            loader = io.load_power
            kw = {'force_index_match':True, 'sum_only':['modes']}
            if columns is not None: kw['columns'] = columns
            Pgal = SpectraSet.from_files(loader, d, basename, coords, dims, args=(mode,), kwargs=kw)
            
            # reindex
            Pgal = self.reindex(Pgal, 'k_cen', self.dk, weights='modes')
            
            # now add errors, using Pmm at z = 0.55 and each galaxy auto spectrum
            Pgal_autos = self.get_Pgal(space=space)
            Pmm = self.get_Pmm(space=space).sel(a='0.6452').values
            auto_names = ['gg', 'cc', 'cAcA', 'cBcB', 'ss', 'sAsA', 'sBsB']
            keys = dict(zip(self.samples, auto_names))
            for k in self.samples:
                this_cross = Pgal.sel(a='0.6452', sample=k).values
                Pgal_auto = Pgal_autos.sel(a='0.6452', sample=keys[k]).values
                utils.add_power_errors(this_cross, Pgal_auto, Pmm)
            
            # set and return
            setattr(self, name, Pgal)
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