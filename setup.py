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
            'compute_fiber_collisions = lsskit.catio.__main__:compute_fiber_collisions',
            'speckmod = lsskit.speckmod.__main__:perform_fit',
            'fit_gp = lsskit.speckmod.__main__:fit_gaussian_process',
            'speckmod_compare = lsskit.speckmod.__main__:compare',
            'speckmod_add_param = lsskit.speckmod.__main__:add_bestfit_param',
            'fit_spline_table = lsskit.speckmod.__main__:fit_spline_table'
        ]
    },
)