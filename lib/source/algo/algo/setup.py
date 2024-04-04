from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("lib/source/algo/cython_batching.pyx"),
    include_dirs=[numpy.get_include()]

)