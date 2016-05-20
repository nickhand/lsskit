import os
import numpy as np
from glob import glob

import xarray as xr
from pyRSD.rsdfit import FittingDriver
from pyRSD.rsdfit.results import EmceeResults, LBFGSResults
from pyRSD.rsdfit.analysis import BestfitParameterSet, to_comparison_table

from lsskit.specksis import utils
from . import fitting_home, rsd_model, model_kmax


def load_results(filename):
    """
    Load a result from file
    """
    try:
        result = EmceeResults.from_npz(filename)
    except:
        result = LBFGSResults.from_npz(filename)
    return result
    
    
class FittingSet(xr.DataArray):
    """
    A subclass of ``xarray.DataArray`` to hold a set of fitting results
    """
    @classmethod
    def from_results(cls, stat, basename, coords, dims, method='median'):
        """
        Initialize the FittingSet from a set of result directories
        """
        if len(dims) != len(coords):
            raise ValueError("shape mismatch between supplied `dims` and `coords`")
        
        data = np.empty(map(len, coords), dtype=object)
        result_dir = os.path.join(fitting_home, stat)
    
        # loop over all the directories
        for i, f in utils.enum_files(result_dir, basename, dims, coords, ignore_missing=False):
            data[i] = FittingResult(stat, f, method=method)
    
        # initialize the base class
        return cls(data, coords=coords, dims=dims)
        
    def plot(self, method='median'):
        """
        Plot the input FittingSet results over one dimension
        """
        import plotify as pfy
        
        if len(self.dims) != 1:
            raise ValueError("exactly one dimension must be specified to plot")
        else:
            dim = self.dims[0]
        
        # loop over the dimension
        for i, val in enumerate(self[dim].values.tolist()):

            # select
            d = self.sel(**{dim:val}).values
        
            # plot
            d.driver.set_fit_results(method=method)
            d.driver.plot() 
            pfy.show()
            
    def table(self, name_formatter, params=None):
        """
        Return a jupyter notebook comparison table
        """
        if len(self.dims) != 1:
            raise ValueError("exactly one dimension must be specified to make comparison table")
        dim = self.dims[0]

        data = []; names = []
        for i, val in enumerate(self[dim].values.tolist()):

            # select
            d = self.sel(**{dim:val}).values
        
            bf = d.bestfit
            names.append(name_formatter(val))
            data.append(bf)
        
        return to_comparison_table(names, data, params=params, fmt='ipynb')
        
    def kdeplot(self, param1, param2, labels=[], cmaps=['Blues', 'Reds', 'Greens'], **kwargs):
        """
        Plot 2D KDE plots for the specified parameters, for a single dimension
        """
        import seaborn as sns
        
        if len(self.dims) != 1:
            raise ValueError("exactly one dimension must be specified to plot")
        else:
            dim = self.dims[0]  
        vals = self[dim].values.tolist()
        
        # check cmap size
        if len(cmaps) and len(vals) > len(cmaps):
            raise ValueError("please specify %d color maps in ``cmaps`` keyword" %len(vals))
        
        # check labels
        if len(labels) and len(labels) != len(vals):
            raise ValueError("please specify %d labels" %len(vals))
        
        # loop over the dimension
        for i, val in enumerate(vals):

            # select
            d = self.sel(**{dim:val}).values
        
            # plot
            if len(cmaps): kwargs['cmap'] = cmaps[i]
            ax = d.result.kdeplot_2d(param1, param2, **kwargs)
        
        # add the legend proxies
        if len(labels):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xcen, ycen = (xmin+xmax)/2, (ymin+ymax)/2
            for i, label in enumerate(labels):
                color = sns.color_palette(cmaps[i])[-2]
                line = sns.plt.fill([xcen, xcen], [ycen, ycen], facecolor=color, label=label, ec='none')
        return ax
    
class FittingResult(object):
    """
    A class to handle reading fitting results
    """
    def __init__(self, stat, fitting_dir, method='median'):
        
        # either 'pkmu' or 'poles'
        self.stat = stat
                
        # the results directory
        self.fitting_dir = fitting_dir
        
        # which fit results to use
        self.method = method
        
    @property
    def result(self):
        """
        The fitting result, either a LBFGSResult or EmceeResult
        """
        try:
            return self._result
        except AttributeError:
            
            # check if combined mcmc result is there
            path = os.path.join(self.fitting_dir, 'info', 'combined_result.npz')
            if os.path.isfile(path):
                r = EmceeResults.from_npz(path)
            else:
                files = glob(os.path.join(self.fitting_dir, '*npz'))
                if not len(files):
                    raise ValueError("no suitable results files found in directory '%s'" %d)
                
                # grab the file modified last
                times = [os.stat(f).st_mtime for f in files]
                try:
                    r = LBFGSResults.from_npz(files[np.argmax(times)])
                except:
                    raise ValueError("if directory is from mcmc fit, define the `info` directory")
                        
            self._result = r
            return self._result
                    
    @property
    def driver(self):
        """
        Return the ``FittingDriver``
        """
        try:
            return self._driver
        except AttributeError:
            d = self.fitting_dir
            r = self.result
            self._driver = FittingDriver.from_directory(d, results_file=r, model_file=rsd_model())
            
            if model_kmax is not None:
                self._driver.theory.model.kmax = model_kmax
            return self._driver
            
    @property
    def fit_type(self):
        """
        Return either 'mcmc' or 'nlopt' depending on the result class
        """
        try:
            return self._fit_type
        except AttributeError:
            
            if isinstance(self.result, EmceeResults):
                self._fit_type = 'mcmc'
            else:
                 self._fit_type = 'nlopt'
            return self._fit_type
            
    @property
    def bestfit(self):
        """
        Return the ``BestfitParameterSet``
        """
        try:
            return self._bestfit
        except AttributeError:
            
            self.driver.set_fit_results(method=self.method)
            meta = {'Np':self.driver.Np, 'Nb':self.driver.Nb, 'min_minus_lkl':-self.driver.lnprob()}
            
            if self.fit_type == 'mcmc':
                self._bestfit = BestfitParameterSet.from_mcmc(self.result, **meta)
            else:
                self._bestfit = BestfitParameterSet.from_nlopt(self.result, **meta)
            
            # scale
            cols = ['best_fit', 'median', 'lower_1sigma', 'upper_1sigma', 
                    'lower_2sigma', 'upper_2sigma', 'gt_1sigma', 'gt_2sigma',
                    'lt_1sigma', 'lt_2sigma']
            if 'NsBsB' in self._bestfit.index:
                self._bestfit.loc['NsBsB', 'scale'] = 1e4
                self._bestfit.loc['NsBsB', cols] /= 1e4
            if 'NcBs' in self._bestfit.index:
                self._bestfit.loc['NcBs', 'scale'] = 1e4
                self._bestfit.loc['NcBs', cols] /= 1e4
            
            return self._bestfit
            