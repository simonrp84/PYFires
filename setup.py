from setuptools import setup, find_packages
from setuptools import setup
from Cython.Distutils import Extension
from Cython.Build import cythonize
import numpy
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

extensions = [
    Extension("pyfires.PYF_WindowStats",
              sources=["pyfires/PYF_WindowStats.pyx"],
              include_dirs=[numpy.get_include()],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
    Extension("pyfires.PYF_Anisotropy",
              sources=["pyfires/PYF_Anisotropy.pyx"],
              include_dirs=[numpy.get_include()],
              define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
    ]

setup(
    name='pyfires',
    version='0.1.0',
    packages=find_packages(include=['PYFires', 'pyfires.*']),

    ext_modules=cythonize(extensions, compiler_directives={'language_level': 3})
)
