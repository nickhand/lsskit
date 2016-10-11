from lsskit import numpy as np
from lsskit.specksis import SpectraSet, HaloSpectraSet, utils, io
from lsskit.data import PowerSpectraLoader
import os

class RunPBHaloPower(PowerSpectraLoader):
    """
    Class to return the halo-related data for the RunPB simulations
    """
    name = "RunPBHaloPower"
    a = ['0.5000', '0.5714', '0.6061', '0.6452', '0.6667', '0.6897', '0.7143', '0.8000', '0.9091', '1.0000']
    mass = list(range(8))
    
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
            d = os.path.join(self.root, 'matter/fourier', space, 'density/power')
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
    # FoF data
    #--------------------------------------------------------------------------
    def get_fof_Phh(self, space='real'):
        """
        Return the FoF Phh in the space specified, either `real` or `redshift`.
        """
        name = '_Phh_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            columns = None
            if space == 'real':
                basename = 'pk_hh{mass}_runPB_%s_{a}.dat' %self.tag
                mode = '1d'
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_hh{mass}_runPB_%s_{a}_Nmu5.dat' %self.tag
                mode = '2d'
            d = os.path.join(self.root, 'halo', space, 'density/power')    
            coords = [self.a, self.mass]
            dims = ['a', 'mass']
            
            # load
            loader = io.load_power
            kw = {'force_index_match':True, 'sum_only':['modes']}
            if columns is not None: kw['columns'] = columns
            Phh = SpectraSet.from_files(loader, d, basename, coords, dims, args=(mode,), kwargs=kw)
            
            # reindex
            Phh = self.reindex(Phh, 'k_cen', self.dk, weights='modes')
            
            # add errors
            Phh.add_power_errors()
            setattr(self, name, Phh)
            return Phh
    
    def get_fof_Phh_cross(self, space='real'):
        """
        Return the Phh cross between different halo bins in the space 
        specified, either `real` or `redshift`
        """
        if space != 'real':
            raise NotImplementedError("only real-space results exist for Phh_cross")
        
        name = '_Phh_x_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo', space, 'density/power')
            basename = 'pk_hh{mass1}{mass2}_runPB_%s_{a}.dat' %self.tag

            coords = [self.a, range(8), range(8)]
            dims = ['a', 'mass1', 'mass2']
            
            # load 
            loader = io.load_power
            kw = {'force_index_match':True, 'sum_only':['modes'], 'columns':['k', 'power', 'modes']}
            Phh = SpectraSet.from_files(loader, d, basename, coords, dims, ignore_missing=True, args=('1d',), kwargs=kw)
            
            # reindex
            Phh = self.reindex(Phh, 'k_cen', self.dk, weights='modes')
            
            # remove NaNs
            Phh = Phh.dropna('mass1', 'all')
            Phh = Phh.dropna('mass2', 'all')
            
            # now add errors, using Pmm at z = 0.55 and each galaxy auto spectrum
            Phh_auto = self.get_fof_Phh(space=space)
            for key, cross in Phh.nditer():
                if cross.isnull(): continue
                this_cross = cross.get()
                Ph1h1 = Phh_auto.sel(a=key['a'], mass=key['mass1']).get()
                Ph2h2 = Phh_auto.sel(a=key['a'], mass=key['mass2']).get()
                utils.add_power_errors(this_cross, Ph1h1, Ph2h2)
                
            setattr(self, name, Phh)
            return Phh
            
    def get_fof_gal_Phh(self, space='real'):
        """
        Return FoF Phh matching the galaxy sample, in the space specified, 
        either `real` or `redshift`
        """
        name = '_Phh_fof_gal_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            columns = None
            if space == 'real':
                basename = 'pk_{sample}_fof_runPB_%s_{a}.dat' %self.tag
                mode = '1d'
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_{sample}_fof_runPB_%s_{a}_Nmu5.dat' %self.tag
                mode = '2d'
            d = os.path.join(self.root, 'halo', space, 'density/power')
                
            samples = ['gg', 'cc', 'cAcA', 'cAcB', 'cBcB', 'cs', 'cAs', 'cBs', 'ss', 'sAsA', 'sAsB', 'sBsB']
            coords = [['0.6452'], samples]
            dims = ['a', 'sample']
            
            # load
            kw = {'sum_only':['modes'], 'force_index_match':True}
            if columns is not None: kw['columns'] = columns
            loader = io.load_power
            Phh = SpectraSet.from_files(loader, d, basename, coords, dims, args=(mode,), kwargs=kw)
            
            # reindex
            Phh = self.reindex(Phh, 'k_cen', self.dk, weights='modes')
            
            # add the errors
            Phh.add_power_errors()
            crosses = {'cAcB':['cAcA', 'cBcB'], 'cs':['cc', 'ss'], 'cAs':['cAcA', 'ss'], 
                        'cBs':['cBcB', 'ss'], 'sAsB':['sAsA', 'sBsB']}
            for x in crosses:
                this_cross = Phh.sel(a='0.6452', sample=x).get()
                k1, k2 = crosses[x]
                a = Phh.sel(a='0.6452', sample=k1).get()
                b = Phh.sel(a='0.6452', sample=k2).get()
                utils.add_power_errors(this_cross, a, b)
                
            setattr(self, name, Phh)
            return Phh
                        
    def get_fof_gal_Phm(self, space='real'):
        """
        Return FoF Phm matching the galaxy sample, in the space specified, 
        either `real` or `redshift`
        """
        if space != 'real':
            raise NotImplementedError("only `real` space results exist form `fof_gal_Phm`")
        
        name = '_Phm_fof_gal_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo-matter/fourier', space, 'power')
            basename = 'pk_{sample}_x_matter_fof_runPB_%s_{a}.dat' %self.tag

            samples = ['gal', 'cen', 'cenA', 'cenB', 'sat', 'satA', 'satB']
            coords = [['0.6452'], samples]
            dims = ['a', 'sample']
            
            # load
            loader = io.load_power
            kw = {'sum_only':['modes'], 'force_index_match':True, 'columns':['k', 'power', 'modes']}
            Pgal = SpectraSet.from_files(loader, d, basename, coords, dims, args=('1d',), kwargs=kw)
            
            # reindex
            Pgal = self.reindex(Pgal, 'k_cen', self.dk, weights='modes')
            
            # now add errors, using Pmm at z = 0.55 and each galaxy auto spectrum
            Pgal_autos = self.get_fof_gal_Phh(space=space)
            Pmm = self.get_Pmm(space=space).sel(a='0.6452').get()
            auto_names = ['gg', 'cc', 'cAcA', 'cBcB', 'ss', 'sAsA', 'sBsB']
            keys = dict(zip(samples, auto_names))
            for k in samples:
                this_cross = Pgal.sel(a='0.6452', sample=k).get()
                Pgal_auto = Pgal_autos.sel(a='0.6452', sample=keys[k]).get()
                utils.add_power_errors(this_cross, Pgal_auto, Pmm)
            
            # set and return
            setattr(self, name, Pgal)
            return Pgal
            
    def get_fof_Phm(self, space='real', average=None):
        """
        Return Phm in the space specified, either `real` or `redshift`
        """
        name = '_Phm_'+space
        if average is not None:
            name += '_'+average
            
        try:
            return getattr(self, name)
        except AttributeError:
            
            columns = None
            if space == 'real':
                basename = 'pk_hm{mass}_runPB_%s_{a}.dat' %self.tag
                mode = '1d'
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_hm{mass}_runPB_%s_{a}_Nmu5.dat' %self.tag
                mode = '2d'
            d = os.path.join(self.root, 'halo-matter/fourier', space, 'power')
                
            coords = [self.a, self.mass]
            dims = ['a', 'mass']
            
            # load
            loader = io.load_power
            kw = {}
            if columns is not None: kw['columns'] = columns
            Phm = SpectraSet.from_files(loader, d, basename, coords, dims, args=(mode,), kwargs=kw)
            
            # reindex
            Phm = self.reindex(Phm, 'k_cen', self.dk, weights='modes')
            
            # average?
            if average is not None:
                Phm = Phm.average(axis=average)
            
            # add errors
            Phm.add_power_errors(self.get_fof_Phh(space), self.get_Pmm(space))
            
            setattr(self, name, Phm)
            return Phm
    
    def get_fof_lambda(self, kind='A', space='real', bias_file=None):
        """
        Return the FoF stochasticity 
        """
        name = '_lambda%s_%s' %(kind, space)
        try:
            return getattr(self, name)
        except AttributeError:

            biases = self.get_fof_halo_biases(bias_file)
            data = HaloSpectraSet(self.get_fof_Phh(space), self.get_fof_Phm(space), self.get_Pmm(space), biases)
            lam = data.to_lambda(kind)
            setattr(self, name, lam)
            return lam
    
    
    def get_fof_lambda_cross(self, kind='A', space='real', bias_file=None):
        """
        Return the stochasticity for Phh_cross
        """
        name = '_lambda%s_x_%s' %(kind, space)
        try:
            return getattr(self, name)
        except AttributeError:

            biases = self.get_fof_halo_biases(bias_file)
            mass_keys = {'mass':['mass1', 'mass2']}
            
            a = self.get_fof_Phh_cross(space)
            b = self.get_fof_Phm(space)
            c = self.get_Pmm(space)
            data = HaloSpectraSet(a, b, c, biases, mass_keys)
            lam = data.to_lambda(kind)
            setattr(self, name, lam)
            return lam
                        
    #--------------------------------------------------------------------------
    # SO data
    #--------------------------------------------------------------------------
    def get_so_Phh(self, space='real'):
        """
        Return SO Phh in the space specified, either `real` or `redshift`
        """
        name = '_Phh_so_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            columns = None
            if space == 'real':
                basename = 'pk_hh{mass}_so_runPB_%s_{a}.dat' %self.tag
                mode = '1d'
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_hh{mass}_so_runPB_%s_{a}_Nmu5.dat' %self.tag
                mode = '2d'
            d = os.path.join(self.root, 'halo', space, 'density/power')
            
            coords = [['0.6452'], self.mass]
            dims = ['a', 'mass']
            
            # load
            loader = io.load_power
            kw = {'force_index_match':True, 'sum_only':['modes']}
            if columns is not None: kw['columns'] = columns
            Phh = SpectraSet.from_files(loader, d, basename, coords, dims, args=(mode,), kwargs=kw)
            
            # reindex
            Phh = self.reindex(Phh, 'k_cen', self.dk, weights='modes')
            
            # add errors
            Phh.add_power_errors()
            
            setattr(self, name, Phh)
            return Phh
    
    def get_so_gal_Phh(self, space='real'):
        """
        Return SO Phh matching the galaxy sample, in the space specified, 
        either `real` or `redshift`
        """
        name = '_Phh_so_gal_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            columns = None
            if space == 'real':
                basename = 'pk_{sample}_so_runPB_%s_{a}.dat' %self.tag
                mode = '1d'
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_{sample}_so_runPB_%s_{a}_Nmu5.dat' %self.tag
                mode = '2d'
            d = os.path.join(self.root, 'halo', space, 'density/power')
            
            samples = ['gg', 'cc', 'cAcA', 'cAcB', 'cBcB', 'cs', 'cAs', 'cBs', 'ss', 'sAsA', 'sAsB', 'sBsB']
            coords = [['0.6452'], samples]
            dims = ['a', 'sample']
            
            # load
            loader = io.load_power
            kw = {'force_index_match':True, 'sum_only':['modes']}
            if columns is not None: kw['columns'] = columns
            Phh = SpectraSet.from_files(loader,d, basename, coords, dims, args=(mode,), kwargs=kw)
            
            # reindex
            Phh = self.reindex(Phh, 'k_cen', self.dk, weights='modes')
            
            # add the errors
            Phh.add_power_errors()
            crosses = {'cAcB':['cAcA', 'cBcB'], 'cs':['cc', 'ss'], 'cAs':['cAcA', 'ss'], 
                        'cBs':['cBcB', 'ss'], 'sAsB':['sAsA', 'sBsB']}
            for x in crosses:
                this_cross = Phh.sel(a='0.6452', sample=x).get()
                k1, k2 = crosses[x]
                a = Phh.sel(a='0.6452', sample=k1).get()
                b = Phh.sel(a='0.6452', sample=k2).get()
                utils.add_power_errors(this_cross, a, b)
            
            setattr(self, name, Phh)
            return Phh
                        
    def get_so_gal_Phm(self, space='real'):
        """
        Return SO Phm matching the galaxy sample, in the space specified, 
        either `real` or `redshift`
        """
        if space != 'real':
            raise NotImplementedError("only `real` space results exist form `so_gal_Phm`")
        
        name = '_Phm_so_gal_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo-matter', space, 'power')
            basename = 'pk_{sample}_x_matter_so_runPB_%s_{a}.dat' %self.tag

            samples = ['gal', 'cen', 'cenA', 'cenB', 'sat', 'satA', 'satB']
            coords = [['0.6452'], samples]
            dims = ['a', 'sample']
            
            # load
            loader = io.load_power
            kw = {'force_index_match':True, 'sum_only':['modes'], 'columns':['k', 'power', 'modes']}
            Phm = SpectraSet.from_files(loader, d, basename, coords, dims, args=('1d',), kwargs=kw)
            
            # reindex
            Phm = self.reindex(Phm, 'k_cen', self.dk, weights='modes')
            
            # now add errors, using Pmm at z = 0.55 and each galaxy auto spectrum
            Phh_autos = self.get_so_gal_Phh(space=space)
            Pmm = self.get_Pmm(space=space).sel(a='0.6452').get()
            
            auto_names = ['gg', 'cc', 'cAcA', 'cBcB', 'ss', 'sAsA', 'sBsB']
            keys = dict(zip(samples, auto_names))
            for k in samples:
                this_cross = Phm.sel(a='0.6452', sample=k).get()
                Phh_auto = Phh_autos.sel(a='0.6452', sample=keys[k]).get()
                utils.add_power_errors(this_cross, Phh_auto, Pmm)
            
            # set and return
            setattr(self, name, Phm)
            return Phm
    
    def get_so_Phm(self, space='real'):
        """
        Return SO Phm in the space specified, either `real` or `redshift`
        """
        name = '_Phm_so_'+space
        try:
            return getattr(self, name)
        except AttributeError:
            
            columns = None
            if space == 'real':
                basename = 'pk_hm{mass}_so_runPB_%s_{a}.dat' %self.tag
                mode = '1d'
                columns = ['k', 'power', 'modes']
            else:
                basename = 'pkmu_hm{mass}_so_runPB_%s_{a}_Nmu5.dat' %self.tag
                mode = '2d'
            d = os.path.join(self.root, 'halo-matter', space, 'power')
                
            coords = [['0.6452'], self.mass]
            dims = ['a', 'mass']
            
            # load
            loader = io.load_power
            kw = {'force_index_match':True, 'sum_only':['modes']}
            if columns is not None: kw['columns'] = columns
            Phm = SpectraSet.from_files(loader, d, basename, coords, dims, args=(mode,), kwargs=kw)
            
            # reindex
            Phm = self.reindex(Phm, 'k_cen', self.dk, weights='modes')
            
            # add errors
            Phm.add_power_errors(self.get_so_Phh(space), self.get_Pmm(space))
            setattr(self, name, Phm)
            return Phm
    
    def get_so_lambda(self, kind='A', space='real', bias_file=None):
        """
        Return the stochasticity 
        """
        name = '_lambda%s_so_%s' %(kind, space)
        try:
            return getattr(self, name)
        except AttributeError:

            biases = self.get_so_halo_biases(bias_file)
            data = HaloSpectraSet(self.get_so_Phh(space), self.get_so_Phm(space), self.get_Pmm(space), biases)
            lam = data.to_lambda(kind)
            setattr(self, name, lam)
            return lam
        
    #--------------------------------------------------------------------------
    # auxiliary data
    #--------------------------------------------------------------------------
    def get_fof_halo_biases(self, kind='original', filename=None):
        """
        Return the linear biases of each halo mass bin 
        """
        try:
            return self._halo_biases
        except:
            if filename is None:
                
                if kind == 'original':
                    filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/biases_halo_mass_bins_original.pickle")
                elif kind == 'new':
                    filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/biases_halo_mass_bins.pickle")
                else:
                    raise ValueError("'kind' should be 'original' or 'new'")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            biases = utils.load_data_from_file(filename, ['a', 'mass'], (len(self.a), len(self.mass)))
            setattr(self, '_halo_biases', biases)
            return biases
                        
    def get_so_halo_biases(self, filename=None):
        """
        Return the SO linear biases of each halo mass bin 
        """
        try:
            return self._so_halo_biases
        except:
            if filename is None:
                filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/biases_so_halo_mass_bins.pickle")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            biases = utils.load_data_from_file(filename, ['a', 'mass'], (1, len(self.mass)))
            setattr(self, '_so_halo_biases', biases)
            return biases
            
    def get_so_gal_halo_biases(self, filename=None):
        """
        Return the SO linear biases of each halo mass bin, matched to the galaxy sample
        """
        try:
            return self._so_gal_halo_biases
        except:
            if filename is None:
                filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/biases_so_gal_halo_mass_bins.pickle")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            biases = utils.load_data_from_file(filename, ['a', 'sample'], (1, 7))
            setattr(self, '_so_gal_halo_biases', biases)
            return biases
            
    def get_fof_gal_halo_biases(self, filename=None):
        """
        Return the FoF linear biases of each halo mass bin, matched to the galaxy sample
        """
        try:
            return self._fof_gal_halo_biases
        except:
            if filename is None:
                filename = os.path.join(os.environ['PROJECTS_DIR'], "RSD-Modeling/RunPBMocks/data/biases_fof_gal_halo_mass_bins.pickle")
                if not os.path.exists(filename):
                    raise ValueError("no file at `%s`, please specify as keyword argument" %filename)
                    
            biases = utils.load_data_from_file(filename, ['a', 'sample'], (1, 7))
            setattr(self, '_fof_gal_halo_biases', biases)
            return biases
    
    def get_fof_halo_masses(self, filename=None):
        """
        Return the average mass of each FoF halo mass bin 
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
            
