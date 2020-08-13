# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='vae_tools',
    version='0.0.1',
    description='Tools to build and play with Variational Autoencoder (VAE)',
    long_description=readme,
    author='Timo Korthals',
    author_email='tkorthals@cit-ec.uni-bielefeld.de',
    url='https://github.com/tik0/vae_tools',
    license=license,
    install_requires=['tensorflow==2.0.2',
                      'tensorboard==2.0.2',
                      'numpy==1.17.4',
                      'tables==3.5.1'
                      'Pillow==5.3.0',
                      'coloredlogs==10.0',
                      'h5py==2.10.0',
                      'scikit-image==0.14.2',
                      'scikit-learn==0.20.0',
                      'pandas==0.17.1',
                      'pycurl==7.43.0',
                      'requests==2.22.0',
                      'scipy==1.2.0',
                      'matplotlib==3.0.3',
                      'GitPython==3.1.0'],
    packages=find_packages(exclude=('tests', 'docs'))
)
