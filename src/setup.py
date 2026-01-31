from setuptools import setup
from Cython.Build import cythonize

setup(
    name='transcripter lib',
    ext_modules=cythonize("transcripter.pyx"),
)
