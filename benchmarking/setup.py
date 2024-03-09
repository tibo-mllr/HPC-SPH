<<<<<<<< HEAD:profiling/setup.py
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "sph_cython.pyx",
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[numpy.get_include()],
)
# python setup.py build_ext --inplace
# python sph.py
========
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        "cythonized/sph_cython.pyx",
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=[numpy.get_include()],
)
>>>>>>>> origin/main:cythonized/setup.py
