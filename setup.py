from setuptools import setup, find_packages
import os

setup(
    name='lsskit',
    version='1.0',
    author='Nick Hand',
    author_email='nicholas.adam.hand@gmail.com',
    packages=find_packages(),
    #scripts=['bin/' + script for script in os.listdir('bin')],
    description='general utilities module for large scale structure analysis',
    entry_points={
        'console_scripts': [
            'compute_multipoles = lsskit.specksis.__main__:compute_multipoles',
            'compare_mcmc_fits = lsskit.specksis.__main__:compare_mcmc_fits',
            'plot_mcmc_bestfit = lsskit.specksis.__main__:plot_mcmc_bestfit',
            'write_data_gaussian_covariance = lsskit.specksis.__main__:write_data_gaussian_covariance',
            'write_covariance = lsskit.specksis.__main__:write_covariance',
            'write_analysis_file = lsskit.specksis.__main__:write_analysis_file',
            'write_analysis_grid = lsskit.specksis.__main__:write_analysis_grid',
            'compute_fourier_biases = lsskit.data.__main__:compute_fourier_biases',
            'compute_config_biases = lsskit.data.__main__:compute_config_biases',
            'save_runPB_galaxy_stats = lsskit.data.__main__:save_runPB_galaxy_stats',
            'compute_fiber_collisions = lsskit.catio.__main__:compute_fiber_collisions',
            'load_mock = lsskit.catio.__main__:load_mock',
            'write_mock_coordinates = lsskit.catio.__main__:write_coordinates',
            'gal_to_halo_samples = lsskit.catio.__main__:gal_to_halo_samples',
            'run_rsdfit = lsskit.rsdfit.run_rsdfit:main'
        ]
    },
)