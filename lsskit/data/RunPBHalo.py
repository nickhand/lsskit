from lsskit import numpy as np
from lsskit.specksis import SpectraSet, HaloSpectraSet, utils
from lsskit.data import PowerSpectraLoader
import os

class RunPBHalo(PowerSpectraLoader):
    """
    Class to return the halo-related data for the RunPB simulations
    """
    name = "RunPBHalo"
    a = ['0.5000', '0.5714', '0.6061', '0.6452', '0.6667', '0.6897', '0.7143', '0.8000', '0.9091', '1.0000']
    mass = range(8)
    
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
    # FoF data
    #--------------------------------------------------------------------------
    def get_fof_Phh(self, space='real'):
        """
        Return the FoF Phh in the space specified, either `real` or `redshift`.
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
            Phh = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'mass']), self.dk)
            Phh.add_errors()
            setattr(self, '_Phh_'+space, Phh)
            return Phh
    
    def get_fof_Phh_cross(self, space='real'):
        """
        Return the Phh cross between different halo bins in the space 
        specified, either `real` or `redshift`
        """
        if space != 'real':
            raise NotImplementedError("only real-space results exist for Phh_cross")
        try:
            return getattr(self, '_Phh_x_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo', space)
            basename = 'pk_hh{mass1}{mass2}_runPB_%s_{a}.dat' %self.tag

            coords = [self.a, range(8), range(8)]
            Phh = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'mass1', 'mass2'], ignore_missing=True), self.dk)
            Phh = Phh.dropna('mass1', 'all')
            Phh = Phh.dropna('mass2', 'all')
            
            # now add errors, using Pmm at z = 0.55 and each galaxy auto spectrum
            Phh_auto = self.get_fof_Phh(space=space)
            for key, cross in Phh.nditer():
                if cross.isnull(): continue
                this_cross = cross.values
                Ph1h1 = Phh_auto.sel(a=key['a'], mass=key['mass1']).values
                Ph2h2 = Phh_auto.sel(a=key['a'], mass=key['mass2']).values
                utils.add_errors(this_cross, Ph1h1, Ph2h2)
                
            setattr(self, '_Phh_x_'+space, Phh)
            return Phh
            
    def get_fof_gal_Phh(self, space='real'):
        """
        Return FoF Phh matching the galaxy sample, in the space specified, either `real` or `redshift`
        """
        try:
            return getattr(self, '_Phh_fof_gal_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo', space)
            if space == 'real':
                basename = 'pk_{sample}_fof_runPB_%s_{a}.dat' %self.tag
            else:
                basename = 'pkmu_{sample}_fof_runPB_%s_{a}_Nmu5.dat' %self.tag
                
            coords = [['0.6452'], ['gg', 'cc', 'cAcA', 'cAcB', 'cBcB', 'cs', 'cAs', 'cBs', 'ss', 'sAsA', 'sAsB', 'sBsB']]
            Phh = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'sample']), self.dk)
            
            # add the errors
            Phh.add_errors()
            crosses = {'cAcB':['cAcA', 'cBcB'], 'cs':['cc', 'ss'], 'cAs':['cAcA', 'ss'], 'cBs':['cBcB', 'ss'], 'sAsB':['sAsA', 'sBsB']}
            for x in crosses:
                this_cross = Phh.sel(a='0.6452', sample=x).values
                k1, k2 = crosses[x]
                utils.add_errors(this_cross, Phh.sel(a='0.6452', sample=k1).values, Phh.sel(a='0.6452', sample=k2).values)
                
            setattr(self, '_Phh_fof_gal_'+space, Phh)
            return Phh
                        
    def get_fof_gal_Phm(self, space='real'):
        """
        Return FoF Phm matching the galaxy sample, in the space specified, either `real` or `redshift`
        """
        if space != 'real':
            raise NotImplementedError("only `real` space results exist form `fof_gal_Phm`")
        try:
            return getattr(self, '_Phm_fof_gal_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo-matter', space)
            basename = 'pk_{sample}_x_matter_fof_runPB_%s_{a}.dat' %self.tag

            samples = ['gal', 'cen', 'cenA', 'cenB', 'sat', 'satA', 'satB']
            coords = [['0.6452'], samples]
            Pgal = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'sample']), self.dk)
            
            # now add errors, using Pmm at z = 0.55 and each galaxy auto spectrum
            Pgal_autos = self.get_fof_gal_Phh(space=space)
            Pmm = self.get_Pmm(space=space).sel(a='0.6452').values
            auto_names = ['gg', 'cc', 'cAcA', 'cBcB', 'ss', 'sAsA', 'sBsB']
            keys = dict(zip(samples, auto_names))
            for k in samples:
                this_cross = Pgal.sel(a='0.6452', sample=k).values
                Pgal_auto = Pgal_autos.sel(a='0.6452', sample=keys[k]).values
                utils.add_errors(this_cross, Pgal_auto, Pmm)
            
            # set and return
            setattr(self, '_Phm_fof_gal_'+space, Pgal)
            return Pgal
            
    def get_fof_Phm(self, space='real'):
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
            Phm = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'mass']), self.dk)
            Phm.add_errors(self.get_fof_Phh(space), self.get_Pmm(space))
            setattr(self, '_Phm_'+space, Phm)
            return Phm
    
    def get_fof_lambda(self, kind='A', space='real', bias_file=None):
        """
        Return the stochasticity 
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
            data = HaloSpectraSet(self.get_fof_Phh_cross(space), self.get_fof_Phm(space), self.get_Pmm(space), biases, mass_keys)
            lam = data.to_lambda(kind)
            setattr(self, name, lam)
            return lam
            
    #--------------------------------------------------------------------------
    # Phh mu components data
    #--------------------------------------------------------------------------
    def get_Phh_mu(self, mu):
        """
        Return the mu0 component of the halo-halo auto-spectrum
        """ 
        allowable = ['mu0', 'mu2', 'mu4', 'mu6']  
        if mu not in allowable:
            raise ValueError('mu must be one of %s' %str(allowable))
            
        name = '_Phh_%s' %mu
        try:
            return getattr(self, name)
        except AttributeError:
            from nbodykit import files, pkresult
            
            d = os.path.join(self.root, 'halo/redshift/poles')
            basename = 'poles_hh{mass}_runPB_mean_{a}.dat'
            coords = [self.mass, self.a]
            dims = ['mass', 'a']
            
            # read in the data
            data = np.empty((len(self.mass), len(self.a)), dtype=object)
            for i, f in utils.enum_files(d, basename, dims, coords, ignore_missing=False):
                
                d, meta = files.ReadPower1DPlainText(f)
                Pshot = meta['volume'] / meta['N1']
                
                # need linear combo of mono + quad
                cols = meta.pop('cols')
                k = d[:,0]; modes = d[:,-1]
                pole_cols = [cols.index(c) for c in ['power_0.real', 'power_2.real', 'power_4.real', 'power_6.real']]
                P0, P2, P4, P6 = d[:,pole_cols].T
            
                if mu == 'mu0':
                    power = P0 - Pshot - 0.5*P2 + 3./8.*P4 - 5./16.*P6
                elif mu == 'mu2':
                    power = 1.5*P2 - 15./4.*P4 + 105./16*P6
                elif mu == 'mu4':
                    power = 35./8.*P4 - 315./16*P6
                else:
                    power = 231./16*P6
                
                d = np.vstack([k, power, modes]).T
                pk = pkresult.PkResult.from_dict(d, ['k', 'power', 'modes'], **meta)
                data[i] = pk

            result = self.reindex(SpectraSet(data, coords=coords, dims=dims), self.dk)
            self._Phh_mu0 = result
            return result
            
    #--------------------------------------------------------------------------
    # SO data
    #--------------------------------------------------------------------------
    def get_so_Phh(self, space='real'):
        """
        Return SO Phh in the space specified, either `real` or `redshift`
        """
        try:
            return getattr(self, '_Phh_so_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo', space)
            if space == 'real':
                basename = 'pk_hh{mass}_so_runPB_%s_{a}.dat' %self.tag
            else:
                basename = 'pkmu_hh{mass}_so_runPB_%s_{a}_Nmu5.dat' %self.tag
                
            coords = [['0.6452'], self.mass]
            Phh = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'mass']), self.dk)
            Phh.add_errors()
            setattr(self, '_Phh_so_'+space, Phh)
            return Phh
    
    def get_so_gal_Phh(self, space='real'):
        """
        Return SO Phh matching the galaxy sample, in the space specified, either `real` or `redshift`
        """
        try:
            return getattr(self, '_Phh_so_gal_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo', space)
            if space == 'real':
                basename = 'pk_{sample}_so_runPB_%s_{a}.dat' %self.tag
            else:
                basename = 'pkmu_{sample}_so_runPB_%s_{a}_Nmu5.dat' %self.tag
                
            coords = [['0.6452'], ['gg', 'cc', 'cAcA', 'cAcB', 'cBcB', 'cs', 'cAs', 'cBs', 'ss', 'sAsA', 'sAsB', 'sBsB']]
            Phh = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'sample']), self.dk)
            
            # add the errors
            Phh.add_errors()
            crosses = {'cAcB':['cAcA', 'cBcB'], 'cs':['cc', 'ss'], 'cAs':['cAcA', 'ss'], 'cBs':['cBcB', 'ss'], 'sAsB':['sAsA', 'sBsB']}
            for x in crosses:
                this_cross = Phh.sel(a='0.6452', sample=x).values
                k1, k2 = crosses[x]
                utils.add_errors(this_cross, Phh.sel(a='0.6452', sample=k1).values, Phh.sel(a='0.6452', sample=k2).values)
            
            setattr(self, '_Phh_so_gal_'+space, Phh)
            return Phh
                        
    def get_so_gal_Phm(self, space='real'):
        """
        Return SO Phm matching the galaxy sample, in the space specified, either `real` or `redshift`
        """
        if space != 'real':
            raise NotImplementedError("only `real` space results exist form `so_gal_Phm`")
        try:
            return getattr(self, '_Phm_so_gal_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo-matter', space)
            basename = 'pk_{sample}_x_matter_so_runPB_%s_{a}.dat' %self.tag

            samples = ['gal', 'cen', 'cenA', 'cenB', 'sat', 'satA', 'satB']
            coords = [['0.6452'], samples]
            Pgal = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'sample']), self.dk)
            
            # now add errors, using Pmm at z = 0.55 and each galaxy auto spectrum
            Pgal_autos = self.get_so_gal_Phh(space=space)
            Pmm = self.get_Pmm(space=space).sel(a='0.6452').values
            auto_names = ['gg', 'cc', 'cAcA', 'cBcB', 'ss', 'sAsA', 'sBsB']
            keys = dict(zip(samples, auto_names))
            for k in samples:
                this_cross = Pgal.sel(a='0.6452', sample=k).values
                Pgal_auto = Pgal_autos.sel(a='0.6452', sample=keys[k]).values
                utils.add_errors(this_cross, Pgal_auto, Pmm)
            
            # set and return
            setattr(self, '_Phm_so_gal_'+space, Pgal)
            return Pgal
    
    
    def get_so_Phm(self, space='real'):
        """
        Return SO Phm in the space specified, either `real` or `redshift`
        """
        try:
            return getattr(self, '_Phm_so_'+space)
        except AttributeError:
            
            d = os.path.join(self.root, 'halo-matter', space)
            if space == 'real':
                basename = 'pk_hm{mass}_so_runPB_%s_{a}.dat' %self.tag
            else:
                basename = 'pkmu_hm{mass}_so_runPB_%s_{a}_Nmu5.dat' %self.tag
                
            coords = [['0.6452'], self.mass]
            Phm = self.reindex(SpectraSet.from_files(d, basename, coords, ['a', 'mass']), self.dk)
            Phm.add_errors(self.get_so_Phh(space), self.get_Pmm(space))
            setattr(self, '_Phm_so_'+space, Phm)
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
    def get_fof_halo_biases(self, filename=None):
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
            
