# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='vae_tools',
    version='0.0.1',
    description='Tools to interact with Variational Autoencoder (VAE)',
    long_description=readme,
    author='Timo Korthals',
    author_email='tkorthals@cit-ec.uni-bielefeld.de',
    url='https://github.com/tik0/vae_tools',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

