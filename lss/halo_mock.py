"""
 halo_mock.py
 lss: subclass of `MockCatalog` to hold a halo mock catalog
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/26/2015
"""
from . import mock_catalog as mc

#-------------------------------------------------------------------------------
class HaloMock(mc.MockCatalog):
    """
    Subclass of `MockCatalog` to represent a halo mock catalog
    """
    def __init__(self, redshift, box_size, units, cosmo=None):
        super(HaloMock, self).__init__(redshift, box_size, units, cosmo=cosmo)
                    
    #---------------------------------------------------------------------------
    # Read-ony properties
    #---------------------------------------------------------------------------
    @property
    def total_halos(self):
        """
        Total number of halos in the mock catalog
        """
        return self._sample_total
        
    #---------------------------------------------------------------------------    
    def restrict_halos(self, halo_condition):
        """
        Restrict the halos included in the current sample by inputing a boolean
        condition in string format, 
        
        ``(key1 == value1)*(key2 == value2) + (key3 ==value3)``
        """
        self.restrictions.set_flags('halo', halo_condition)
        self._sample = self.restrictions.slice_frame(self._data)
          
    #---------------------------------------------------------------------------
    def load(self, filename, info_dict, halo_id, skip_lines=0):
        """
        Load objects from a file
        
        Parameters
        ----------
        filename : str
            The name of the file containing the info to load into each halo
        info_dict : dict
            A dictionary with keys corresponding to the names of the columns
            in the `DataFrame` and the values corresponding to the column 
            numbers to read from the input file
        halo_id : str
            The name of the column defining the halo identifier                 
        skip_lines : int
            The number of lines to skip when reading the input file; 
            default is 0
        """
        kwargs               = {}
        kwargs['extra_info'] = {'halo_id' : halo_id}
        kwargs['skip_lines'] = skip_lines  
        mc.MockCatalog.load(self, filename, info_dict, **kwargs)
            
    #---------------------------------------------------------------------------
    def plot_mass_distribution(self, mass_col):
        """
        Plot the mass distribution of all galaxies, and satellites only
        """
        import plotify as pfy
        
        # halo masses for all halos
        masses = mc.np.asarray(self.sample[mass_col])
        bins, pdf, dM = self._compute_pdf(masses, log=True)

        # plot
        ax = pfy.gca()
        pfy.bar(bins[:-1], pdf, width=dM, bottom=pdf*0., color=ax.color_list[0], alpha=0.5, grid='y')

        # make it look nice
        pfy.plt.subplots_adjust(bottom=0.15)
        ax.x_log_scale()
        ax.ylabel.update("d$p$ / d $\mathrm{log_{10}}M$", fontsize=16)
        if self.units == 'relative':
            mass_units = "h^{-1} M_\odot"
        else:
            mass_units = "M_\odot"
        ax.xlabel.update(r"$M_\mathrm{halo} \ (%s)$" %mass_units, fontsize=16)

        return ax        
    #---------------------------------------------------------------------------
#-------------------------------------------------------------------------------


        
