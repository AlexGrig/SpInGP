#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup for cython 
"""
# gcc ./pardiso_unsym.c -o pardiso_unsym -L /home/alex/Downloads 
#    -lpardiso500-GNU481-X86-64 -lblas -llapack -lgfortran -fopenmp -lpthread -lm

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("pardiso_bind", ["pardiso_bind.pyx"],
#        include_dirs = [...],
        extra_compile_args = ['-pthread','-fopenmp'],
        libraries = ['pardiso500-GNU481-X86-64','blas','lapack','gfortran','pthread'],
        library_dirs = ['/home/alex/Downloads']), ]

setup(
    name = "pardiso_bind",
    ext_modules = cythonize(extensions),
)