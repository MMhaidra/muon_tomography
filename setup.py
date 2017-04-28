from distutils.core import setup
from Cython.Build import cythonize

setup(
        name='msc app',
        ext_modules=cythonize('*.py'),
)
