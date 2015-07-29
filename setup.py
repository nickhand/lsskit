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
            'speckmod = lsskit.speckmod.__main__:perform_fit'
        ]
    },
)