from setuptools import setup, find_packages

setup(
    name='pyfires',
    version='0.1.0',
    packages=find_packages(include=['PYFires', 'pyfires.*'])
)