# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 16:56:12 2017

@author: alex
"""

#DEF "NPY_NO_DEPRECATED_API" 
#DEF "NPY_1_7_API_VERSION" 

from libc.stdio cimport printf, sscanf
from libc.stdlib cimport getenv
#cimport libc.math
from libc.stdlib cimport malloc, free 

import numpy as np
cimport numpy as np
cimport cython
#/* PARDISO prototype. */
#name = "solve_sym_pardiso"
