from lsskit.data import PowerSpectraLoader
from lsskit.specksis import SpectraSet, HaloSpectraSet
from lsskit import numpy as np
from pyRSD import data as sim_data
from nbodykit.dataset import DataSet
import os

def make_kedges(data):

    k = data[:,0]
    kedges = np.empty(len(k)+1)
    dk = 0.5*np.diff(k)
    kedges[0], kedges[-1] = k[0]-dk[0], k[-1]+dk[-1]
    kedges[1:-1] = k[1:]-dk
    return kedges

def load_data(data, space, **meta):
    """
    Load the data, saved in columns to an ASCII file, into either
    a DataSet
    """
    # make the edges
    k = data[:,0]
    kedges = make_kedges(data)
    factor = 1./12**0.5

    if space == 'redshift':
        Nmu = 5
        muedges = np.linspace(0, 1, Nmu+1)
        mu = 0.5*(muedges[1:] + muedges[:-1])
        k, mu = np.broadcast_arrays(k[...,None], mu[None,...])
        power = np.vstack([data[:,2*i+1] for i in range(Nmu)]).T
        error = np.vstack([data[:,2*(i+1)] for i in range(Nmu)]).T
        
        dtype = np.dtype([('k','f8'), ('mu','f8'), ('power','f8'), ('error','f8')])
        d = numpy.empty(k.shape, dtype=dtype)
        d['k'] = k; d['mu'] = mu; d['power'] = power; d['error'] = error*factor
        return DataSet(['k', 'mu'], [kedges, muedges],d, **meta)
    
    elif space == 'real':
        dtype = np.dtype([('k','f8'), ('power','f8'), ('error','f8')])
        d = numpy.empty(len(k), dtype=dtype)
        d['k'] = k; d['power'] = data[:,-2]; d['error'] = data[:,-1]*factor
        return DataSet(['k'], [kedges], d, **meta)
    
    else:
        raise ValueError("cannot load data; space should be 'redshift' or 'real'")


class TeppeiSimsPower(PowerSpectraLoader):
    name = "TeppeiSimsPower"
    z = ['0.000', '0.509', '0.989']
    mass = range(4)

    def __init__(self, root):
        self.root = root

    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)

    #--------------------------------------------------------------------------
    # data accessors
    #--------------------------------------------------------------------------
    def get_Phh(self, space='real'):
        """
        Return Phh in the space specified. Only 'real' is currently implemented.
        """
        try:
            return getattr(self, '_Phh_'+space)
        except AttributeError:

            d = os.path.join(self.root, 'halo')
            mass_bins = ["00020_00060", "00060_00180", "00180_00540", "00540_01620"]
            redshifts = ['z007', 'z005', 'z004']
            Pshot = [[2336., 6494., 21882., 101112.], [2849., 9174., 41152., 314465.], [4065., 16447., 111359., np.nan]]
            meta = {'box_size' : 1600., 'N1':0, 'N2':0}
            basename = 'pkmu_chi_00_h_h_{mass}_{z}_1-3_02-13binaaQ'

            data = np.empty((len(redshifts), len(mass_bins)), dtype=object)
            for i in range(len(redshifts)):
                for j in range(len(mass_bins)):
                    args = {'mass':mass_bins[j], 'z':redshifts[i]}
                    meta['N1'] = meta['N2'] = meta['box_size']**3 / Pshot[i][j]

                    filename = os.path.join(d, basename.format(**args))
                    if os.path.exists(filename):
                        data[i,j] = load_data(np.loadtxt(filename), space, **meta)
                    else:
                        data[i,j] = np.nan

            Phh = SpectraSet(data, coords=[self.z, self.mass], dims=['z', 'mass'])
            setattr(self, '_Phh_'+space, Phh)
            return Phh

    def get_Pmm(self, space='real'):
        """
        Return Pmm in the space specified, either `real` or `redshift`
        """
        try:
            return getattr(self, '_Pmm_'+space)
        except AttributeError:

            d = os.path.join(self.root, 'dark_matter')
            redshifts = ['z007', 'z005', 'z004']
            meta = {'box_size' : 1600., 'N1':np.inf, 'N2':np.inf}
            basename = 'pkmu_chi_00_m_m_{z}_1-3_02-13binaaQ'

            data = np.empty(len(redshifts), dtype=object)
            for i in range(len(redshifts)):
                filename = os.path.join(d, basename.format(z=redshifts[i]))
                if os.path.exists(filename):
                    data[i] = load_data(np.loadtxt(filename), space, **meta)
                else:
                    data[i] = np.nan

            Pmm = SpectraSet(data, coords=[self.z], dims=['z'])
            setattr(self, '_Pmm_'+space, Pmm)
            return Pmm

    def get_Phm(self, space='real'):
        """
        Return Phm in the space specified, either `real` or `redshift`
        """
        if space != 'real':
            raise NotImplementedError("sorry, only real-space results exist for Phm")
        try:
            return getattr(self, '_Phm_'+space)
        except AttributeError:

            d = os.path.join(self.root, 'halo-matter')
            mass_bins = ["00020_00060", "00060_00180", "00180_00540", "00540_01620"]
            redshifts = ['z007', 'z005', 'z004']
            meta = {'box_size' : 1600., 'N1':np.inf, 'N2':np.inf}
            basename = 'pk_mr0_hr0_{mass}_{z}_linaax'

            data = np.empty((len(redshifts), len(mass_bins)), dtype=object)
            for i in range(len(redshifts)):
                for j in range(len(mass_bins)):
                    args = {'mass':mass_bins[j], 'z':redshifts[i]}
                    filename = os.path.join(d, basename.format(**args))
                    if os.path.exists(filename):
                        data[i,j] = load_data(np.loadtxt(filename), space, **meta)
                    else:
                        data[i,j] = np.nan

            Phm = SpectraSet(data, coords=[self.z, self.mass], dims=['z', 'mass'])
            setattr(self, '_Phm_'+space, Phm)
            return Phm

    def get_lambda(self, kind='A', space='real'):
        """
        Return the stochasticity
        """
        if space != 'real':
            raise NotImplementedError("sorry, only real-space results exist for Lambda")

        name = '_lambda%s_%s' %(kind, space)
        try:
            return getattr(self, name)
        except AttributeError:

            biases = self.get_halo_biases().dropna('mass')
            Phh = self.get_Phh(space).dropna('mass')
            Phm = self.get_Phm(space).dropna('mass')
            data = HaloSpectraSet(Phh, Phm, self.get_Pmm(space), biases)
            lam = data.to_lambda(kind)
            setattr(self, name, lam)
            return lam


    #--------------------------------------------------------------------------
    # DM correlators
    #--------------------------------------------------------------------------
    def get_P01(self, smooth=False):
        """
        Return the dark matter P01[mu^2] correlator in real-space
        """
        tag = '_smooth' if smooth else ''
        name = '_P01%s' %tag
        try:
            return getattr(self, name)
        except AttributeError:

            redshifts = ['z007', 'z005', 'z004']
            meta = {'box_size' : 1600., 'N1':np.inf, 'N2':np.inf}
            if smooth:
                from pyRSD.data import P01_mu2_z_0_000, P01_mu2_z_0_509, P01_mu2_z_0_989
                smooth_data = [P01_mu2_z_0_000(), P01_mu2_z_0_509(), P01_mu2_z_0_989()]
            else:
                d = os.path.join(self.root, 'dark_matter')
                basename = 'pkmu_chi_01_m_m_{z}_1-3_02-13binaaQ'

            data = np.empty(len(redshifts), dtype=object)

            for i in range(len(redshifts)):
                if not smooth:
                    filename = os.path.join(d, basename.format(z=redshifts[i]))
                    if os.path.exists(filename):
                        x = np.loadtxt(filename)
                    else:
                        data[i] = np.nan
                        continue
                else:
                    x = smooth_data[i]
                kedges = make_kedges(x)

                data_dict = {}
                data_dict['k'] = x[:,0]
                if not smooth:
                    data_dict['power'] = x[:,-2]*2*x[:,0]
                    data_dict['error'] = x[:,-1]*2*x[:,0]
                else:
                    data_dict['power'] = x[:,1]
                data[i] = Power1dDataSet(kedges, data_dict, **meta)

            toret = SpectraSet(data, coords=[self.z], dims=['z'])
            setattr(self, name, toret)
            return toret

    def get_P11(self, mu, smooth=False):
        """
        Return the dark matter P11[mu^2] or P11[mu^4] correlator in real-space
        """
        tag = '_smooth' if smooth else ''
        name = '_P11_%s%s' %(mu, smooth)
        if mu not in ['mu2', 'mu4']:
            raise ValueError('`mu` must be one of [`mu2`, `mu4`]')

        try:
            return getattr(self, name)
        except AttributeError:

            redshifts = ['z007', 'z005', 'z004']
            meta = {'box_size' : 1600., 'N1':np.inf, 'N2':np.inf}
            if smooth:
                if mu == 'mu2':
                    from pyRSD.data import P11_mu2_z_0_000, P11_mu2_z_0_509, P11_mu2_z_0_989
                    smooth_data = [P11_mu2_z_0_000(), P11_mu2_z_0_509(), P11_mu2_z_0_989()]
                else:
                    from pyRSD.data import P11_mu4_z_0_000, P11_mu4_z_0_509, P11_mu4_z_0_989
                    smooth_data = [P11_mu4_z_0_000(), P11_mu4_z_0_509(), P11_mu4_z_0_989()]
            else:
                d = os.path.join(self.root, 'dark_matter')
                basename = 'pkmu_chi_11_m_m_{z}_1-3_02-13binaaQ'

            if mu == 'mu2':
                cols = [-4, -2]
            else:
                cols = [-3, -1]

            data = np.empty(len(redshifts), dtype=object)
            for i in range(len(redshifts)):
                if not smooth:
                    filename = os.path.join(d, basename.format(z=redshifts[i]))
                    if os.path.exists(filename):
                        x = np.loadtxt(filename)
                    else:
                        data[i] = np.nan
                        continue
                else:
                    x = smooth_data[i]
                kedges = make_kedges(x)

                data_dict = {}
                data_dict['k'] = x[:,0]
                if not smooth:
                    data_dict['power'] = x[:,cols[0]]*x[:,0]**2
                    data_dict['error'] = x[:,cols[1]]*x[:,0]**2
                else:
                    data_dict['power'] = x[:,1]
                data[i] = Power1dDataSet(kedges, data_dict, **meta)

            toret = SpectraSet(data, coords=[self.z], dims=['z'])
            setattr(self, name, toret)
            return toret

    def get_Pdv(self, smooth=False):
        """
        Return the dark matter Pdv cross-spectrum in real-space
        """
        tag = '_smooth' if smooth else ''
        name = '_P11%s' %smooth
        try:
            return getattr(self, name)
        except AttributeError:

            redshifts = ['z007', 'z005', 'z004']
            meta = {'box_size' : 1600., 'N1':np.inf, 'N2':np.inf}
            if smooth:
                from pyRSD.data import Pdv_mu0_z_0_000, Pdv_mu0_z_0_509, Pdv_mu0_z_0_989
                smooth_data = [Pdv_mu0_z_0_000(), Pdv_mu0_z_0_509(), Pdv_mu0_z_0_989()]
            else:
                d = os.path.join(self.root, 'dark_matter')
                basename = 'pkmu_chi_01_m_m_{z}_1-3_02-13binvvQ'

            data = np.empty(len(redshifts), dtype=object)
            for i in range(len(redshifts)):
                if not smooth:
                    filename = os.path.join(d, basename.format(z=redshifts[i]))
                    if os.path.exists(filename):
                        x = np.loadtxt(filename)
                    else:
                        data[i] = np.nan
                        continue
                else:
                    x = smooth_data[i]
                kedges = make_kedges(x)

                data_dict = {}
                data_dict['k'] = x[:,0]
                if not smooth:
                    data_dict['power'] = x[:,-2]*(-x[:,0])
                    data_dict['error'] = x[:,-1]*(x[:,0])
                else:
                    data_dict['power'] = x[:,1]
                data[i] = Power1dDataSet(kedges, data_dict, **meta)

            toret = SpectraSet(data, coords=[self.z], dims=['z'])
            setattr(self, name, toret)
            return toret

    #--------------------------------------------------------------------------
    # galaxy spectra
    #--------------------------------------------------------------------------
    def get_Pgal(self, space='real'):
        """
        Return galaxy component spectra
        """
        try:
            return getattr(self, '_Pgal_'+space)
        except AttributeError:

            samples = ['gg', 'cc', 'cAs', 'sAsA', 'sAsB']
            meta = {'box_size' : 1600., 'N1':np.inf, 'N2':np.inf}
            basename = 'P{sample}_z_0_509'

            data = np.empty(len(samples), dtype=object)
            for i in range(len(samples)):

                d = getattr(sim_data, basename.format(sample=samples[i]))()
                if space == 'real' and d.shape[-1] == 13 or space == 'redshift':
                    data[i] = load_data(d, space, **meta)
                else:
                    data[i] = np.nan

            Pgal = SpectraSet(data, coords=[samples], dims=['sample'])
            setattr(self, '_Pgal_'+space, Pgal)
            return Pgal

    def get_Pgal_no_fog(self):
        """
        Return galaxy component spectra in redshift space, with no FOG effect
        """
        try:
            return getattr(self, '_Pgal_no_fog')
        except AttributeError:

            samples = ['cAs', 'sAsA', 'sAsB']
            meta = {'box_size' : 1600., 'N1':np.inf, 'N2':np.inf}
            basename = 'P{sample}_no_fog_z_0_509'

            data = np.empty(len(samples), dtype=object)
            for i in range(len(samples)):

                d = getattr(sim_data, basename.format(sample=samples[i]))()
                data[i] = load_data(d, 'redshift', **meta)

            Pgal = SpectraSet(data, coords=[samples], dims=['sample'])
            setattr(self, '_Pgal_no_fog', Pgal)
            return Pgal

    def get_galaxy_poles(self):
        """
        Return galaxy multipoles in redshift space
        """
        try:
            return getattr(self, '_poles')
        except AttributeError:

            samples = ['mono', 'quad']
            meta = {'box_size' : 1600., 'N1':np.inf, 'N2':np.inf}
            basename = 'Pgg_{sample}_z_0_509'

            data = np.empty(len(samples), dtype=object)
            for i in range(len(samples)):
                d = getattr(sim_data, basename.format(sample=samples[i]))()
                data[i] = load_data(d, 'real', **meta)

            Pgal = SpectraSet(data, coords=[samples], dims=['sample'])
            setattr(self, '_poles', Pgal)
            return Pgal

    def get_gal_biases(self, filename=None):
        """
        Return the linear biases of each galaxy sample
        """
        try:
            return self._gal_biases
        except:
            import xarray as xr
            samples = ['cen', 'cenA', 'cenB', 'gal', 'sat', 'satA', 'satB']
            data = [2.02, 1.91, 2.92, 2.17, 3.26, 2.68, 4.00]
            biases = xr.DataArray(data, coords=[samples], dims=['sample'])
            setattr(self, '_gal_biases', biases)
            return biases

    def get_gal_stats(self):
        """
        Return a dictionary holding the galaxy sample statistics, fractions, etc
        """
        try:
            return self._gal_stats
        except:
            toret = {}
            toret['mean'] = {'N_tot':1.5e7, 'fs':0.123, 'fsB':0.432, 'fcB':0.104, 'NcBs':1./2.77e-5, 'NsBsB':1.19e5}
            setattr(self, '_gal_stats', toret)
            return toret


    def get_halo_biases(self):
        """
        Return the linear biases of each halo mass bin
        """
        try:
            return self._halo_biases
        except:
            import xarray as xr
            data = [[1.17, 1.46, 2.03, 3.04], [1.64, 2.18, 3.13, 4.82], [2.33, 3.18, 4.72, np.nan]]
            biases = xr.DataArray(data, coords=[self.z, self.mass], dims=['z', 'mass'])
            setattr(self, '_halo_biases', biases)
            return biases
