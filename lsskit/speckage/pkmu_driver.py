"""
 pkmu_driver.py
 this contains functions for wrapping Pat's ``measure_and_fit_discrete.out``
 code in Python
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 07/19/2014
"""
import subprocess 
import os
from . import hod_mock, tools
from utils import pytools
from cosmology.utils.units import h_conversion_factor

#-------------------------------------------------------------------------------
def compute_PB_Pkmu(infile, options={}, show_help=False, stdout=None):
    """
    Call Pat McDonald's BOSS cosmology code ``measure_and_fit_discrete.out`` 
    to compute P(k, mu) for a set of discrete points in a periodic box. 
    
    Notes
    -----
    Assumes the code executable lives in ``$COSMOLOGYINC/ComovingPower/Examples/``
    
    Parameters
    ----------
    infile : str
        The name of the file holding the coordinates
    """
    # get the path of the executable
    executable = "%s/ComovingPower/Examples/measure_and_fit_discrete.out" %os.environ['COSMOLOGYINC']
    
    # make sure it exists
    if not os.path.exists(executable):
        raise ValueError("Executable for BOSS cosmology P(k, mu) code not found at %s" %executable)
        
    
    # only show the help
    if show_help:
        
        retcode = subprocess.call([executable, "-h"])
        if retcode:
            raise ValueError("Error calling P(k, mu) code")
            
    else:
        
        # get the absolute path name for the object_filename
        if infile is not None:
            infile_abs = os.path.abspath(infile)
            calling_signature = [executable, "--infile", "%s" %infile_abs]
        else:
            calling_signature = [executable]
            
        for k, v in options.items():
            calling_signature += ["--%s" %k, str(v)]
    
        # call the code
        if stdout is not None:
            retcode = subprocess.call(calling_signature, stdout=stdout, stderr=stdout)
        else:
            retcode = subprocess.call(calling_signature)
        #if retcode:
        #    raise ValueError("Error calling P(k, mu) code")
           
#end compute_PB_Pkmu

#-------------------------------------------------------------------------------
def compute_PB_poles(infile, options={}, show_help=False, stdout=None):
    """
    Call Pat McDonald's BOSS cosmology code ``fit_poles.out`` 
    to fit multipoles to a given P(k, mu) measurement
    
    Notes
    -----
    Assumes the code executable lives in ``$COSMOLOGYINC/ComovingPower/Examples/``
    
    Parameters
    ----------
    infile : str
        The name of the file holding the coordinates
    """
    # get the path of the executable
    executable = "%s/ComovingPower/Examples/fit_poles.out" %os.environ['COSMOLOGYINC']
    
    # make sure it exists
    if not os.path.exists(executable):
        raise ValueError("Executable for BOSS cosmology multipoles code not found at %s" %executable)
        
    
    # only show the help
    if show_help:
        
        retcode = subprocess.call([executable, "-h"])
        if retcode:
            raise ValueError("Error calling multipoles code")
            
    else:
        
        # get the absolute path name for the object_filename
        if infile is not None:
            infile_abs = os.path.abspath(infile)
            calling_signature = [executable, "--CPM_default_filename", "%s" %infile_abs]
        else:
            calling_signature = [executable]
            
        for k, v in options.items():
            calling_signature += ["--%s" %k, str(v)]
    
        # call the code
        if stdout is not None:
            retcode = subprocess.call(calling_signature, stdout=stdout, stderr=stdout)
        else:
            retcode = subprocess.call(calling_signature)
        #if retcode:
        #    raise ValueError("Error calling P(k, mu) code")
           
#end compute_PB_poles

#-------------------------------------------------------------------------------
def fit_bias(params, mock):
    """
    Fit the large-scale bias
    """ 
    import tempfile
    import shutil
    
    params = params.copy()
    
    # get the name of the output file
    output_dir = 'power_fits'
    with pytools.ignored(OSError):
        os.makedirs(output_dir)
    output_file = '%s/%s_fitTSAL.dat' %(output_dir, params['output_tag'])
    
    # if the fit file doesn't exist, we need to make it
    if not os.path.exists(output_file):
        
        # the kmax to compute P(k, mu) to 
        kmax = params.pop('kmax')

        # first clear any restrictions
        mock.clear_restrictions()
        
        # set the restrictions
        if params['galaxy_restrict'] is not None:
            mock.restrict_galaxies(params['galaxy_restrict'])

        if params['halo_restrict'] is not None:
            mock.restrict_halos(params['halo_restrict'])

        # compute the shot noise for this sample in (Mpc)^3
        output_units_factor = h_conversion_factor('volume', mock.units, params['file_units'], mock.cosmo['h'])
        volume = mock.box_size**3
        Pshot = volume / mock.total_galaxies * output_units_factor
        
        # set the missing parameters
        params['redshift_space'] = False
        params['object_file'] = None
    
        # temporary file to hold the parameters
        temp_param_file = tempfile.NamedTemporaryFile()
        params.writeToFile(temp_param_file.name)
        
        # now call compute_Pkmu
        options = "--log --no_package --Ncells 128 --kmax %f --k_order 0 --kpar_order 0 --vary_noise 0 --shot_noise %.0f --dk 0.005 --kmin 0.005 " %(kmax, Pshot)
        ans = os.system("compute_Pkmu %s %s" %(temp_param_file.name, options))
        if ans:
            raise ValueError("Cannot compute bias; error in `compute_Pkmu`")
        
        # now package the results
        with pytools.ignored(OSError):
            os.remove('comovingPDT.pdf')
        with pytools.ignored(OSError):
            os.remove('Pdtbpa.cbpm')
        with pytools.ignored(OSError):
            os.rename('cPfitAnalysisTSAL.txt', output_file)
        with pytools.ignored(OSError):
            shutil.rmtree('logs')   
            
    return tools.extract_bias(output_file)
#end fit_bias

#-------------------------------------------------------------------------------


    


    
           
    
    

