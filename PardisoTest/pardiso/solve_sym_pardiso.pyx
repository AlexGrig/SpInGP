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

cimport cython
import numpy as np
cimport numpy as np

from sys import exit

cdef extern from "numpy/arrayobject.h":
        void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
        
#/* PARDISO prototype. */
#name = "solve_sym_pardiso"

cdef:
    extern void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
    extern void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                             double *, int    *,    int *, int *,   int *, int *,
                             int *, double *, double *, int *, double *);
    extern void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
    extern void pardiso_chkvec     (int *, int *, double *, int *);
    extern void pardiso_printstats (int *, int *, double *, int *, int *, int *, double *, int *);

# The function solves linear symmetrix system in Pardiso library
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cpdef np.ndarray[double, ndim=1, mode="c"] solve_linear_system(np.ndarray[double, ndim=1, mode="c"] p_a, # (A-csr matrix) A.data
#                             np.ndarray[int, ndim=1, mode="c"] p_ia, # (A-csr matrix) A.indptr
#                             np.ndarray[int, ndim=1, mode="c"] p_ja, # A.indices
#                             np.ndarray[double, ndim=1, mode="c"] p_rhs):

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple solve_linear_system(double[:] p_a, # (A-csr matrix) A.data
                             int[:] p_ia, # (A-csr matrix) A.indptr
                             int[:] p_ja, # A.indices
                             double[:] p_rhs,
                             int supernode_size,
                             bint comp_determinant,
                             bint comp_selective_inverse):    
    # If supernode size < 0, then it is ot used.
    cdef:
         #/* Matrix data. */
        int    n = (p_ia.shape[0]-1)
    
        double* a = &p_a[0]
        
        int* ia = &p_ia[0]
        int* ja = &p_ja[0]       
        
        int      nnz = p_ja.shape[0]  #ia[n];
        int      mtype = -2;        #/* Real symmetric matrix */
    
        #/* RHS and solution vectors. */
        double *b = &p_rhs[0],
 
        double *x = <double*> malloc(sizeof(double) * n)
        # double[:] x = np.empty(n, dtype=np.float)

        int      nrhs = 1;          #/* Number of right hand sides. */
    
        #/* Internal solver memory pointer pt,                  */
        #/* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
        #/* or void *pt[64] should be OK on both architectures  */ 
        void    *pt[64]; 
    
        #/* Pardiso control parameters. */
        int      iparm[64];
        double   dparm[64]; 
        double log_determ=-1;
        int      maxfct, mnum, phase, error, msglvl, solver;
    
        #/* Number of processors. */
        int      num_procs;
    
        #/* Auxiliary variables. */
        char    *var;
        int      i, k;
    
        double   ddum;              #/* Double dummy */
        int      idum;              #/* Integer dummy. */
        
    error = 0;
    solver = 0; #/* use sparse direct solver */

    iparm[1] = 0; # different reordering algorithm

    pardisoinit(pt,  &mtype, &solver, iparm, dparm, &error);
    
    if (error != 0):
        if (error == -10 ):
            printf("No license file found \n");
        if (error == -11 ):
            printf("License is expired \n");
        if (error == -12 ):
            printf("Wrong username or hostname \n");
    else:
        printf("[PARDISO]: License check was successful ... \n");
    
    var = getenv("OMP_NUM_THREADS");
    if(var != NULL):
        sscanf( var, "%d", &num_procs );
    else:
        printf("Set environment OMP_NUM_THREADS to 1");
        exit(1);
    
#    # ! TODO move these to the python level to use parallel
#    # Convert indices to fortran based:
#    for i in range(n+1):
#        ia[i] += 1;
#
#    for i in range(nnz):
#        ja[i] += 1;


    iparm[2]  = num_procs;

    maxfct = 1;		#/* Maximum number of numerical factorizations.  */
    mnum   = 1;         #/* Which factorization to use. */
    
    msglvl = 1;         #/* Print statistical information  */
    error  = 0;         #/* Initialize error flag */


#    # Debug ->
#    pardiso_chkmatrix(&mtype, &n, a, ia, ja, &error);
#    if (error != 0):
#        printf("\nERROR in consistency of matrix: %d", error);
#        exit(1);
#    
#    pardiso_chkvec(&n, &nrhs, b, &error);
#    if (error != 0):
#        printf("\nERROR  in right hand side: %d", error);
#        exit(1);
#        
#    pardiso_printstats(&mtype, &n, a, ia, ja, &nrhs, b, &error);
#    if (error != 0):
#        printf("\nERROR right hand side: %d", error) weg;
#        exit(1);
#    # Debug <-

    if supernode_size > 0:
        iparm[29] = supernode_size; #/* set size of the supernode*/
        
    # Symbolic factorization
    phase = 11;
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
    
    # Numerical factorization
    phase = 22;
    
    if comp_determinant:
        iparm[32] = 1; #/* compute determinant */
    
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error, dparm);
    
    if (error != 0):
        printf("\nERROR during numerical factorization: %d", error);
        exit(2);
        
    printf("\nFactorization completed ...\n ");
    
    #Solve
    phase = 33;
    iparm[7]  = 1;       #/* Max numbers of iterative refinement steps. */
    
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);
    
    if (error != 0):
        printf("\nERROR during solution: %d", error);
        exit(3);
        
    if comp_determinant:
        log_determ = dparm[32];
        
    cdef double[:,] solution =  <double[:n,]> x;  
    
    printf("\nSize of the SUPERNODE: %d", iparm[29]);
    
    #cdef double[:,] solution =  <double[:n,]> x;
    
#    * -------------------------------------------------------------------- */    
# /* ... Inverse factorization.                                           */                                       
# /* -------------------------------------------------------------------- */
#    cdef double[:,] a_inv;
#    cdef int[:,] ia_inv;
#    cdef int[:,] ja_inv;
    
    if comp_selective_inverse:
        printf("\nCompute Diagonal Elements of the inverse of A ... \n");
        phase = -22;
        iparm[35]  = 1; #/*  no not overwrite internal factor L */ 
        #iparm[36]  = 1; #/*  return the selected inversioni in  in full symmetric triangular CSR format*/ 
        pardiso (pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, &idum, &nrhs,
                 iparm, &msglvl, b, x, &error,  dparm);
     
    
        a_inv =  <double[:nnz,]> a;



#        ia_inv =  <int[:n+1,]> ia; # copying
#        ja_inv =  <int[:nnz,]> ja; # copying
    


#       /* print diagonal elements */
#       for (k = 0; k < n; k++)
#       {
#            int j = ia[k]-1;
#            /* printf("Diagonal element of A^{-1} = %d %d %32.24e\n", k, ja[j]-1, a[j]); */
#			printf ("Diagonal element of A^{-1} = %32.24e =  %32.24e \n", a[j], diag[k]);
#       }


    
    #    #/* -------------------------------------------------------------------- */
    #/* ..  Convert matrix back to 0-based C-notation.                       */
    #/* -------------------------------------------------------------------- */
    
#    # TODO: are these needed? Yes, but move to python for paralellization.
#    #for (i = 0; i < n+1; i++) {
#    for i in range(n+1):
#        ia[i] -= 1;
#    
#    #for (i = 0; i < nnz; i++) {
#    for i in range(nnz):
#        ja[i] -= 1;
#    
       
    #/* -------------------------------------------------------------------- */
    #/* ..  Termination and release of memory.                               */
    #/* -------------------------------------------------------------------- */
    phase = -1;                 #/* Release internal memory. */
    
    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error,  dparm);
    
    printf("\nLast pardiso bind. n=%d !\n", nnz);
#    cdef np.ndarray[np.float_t, ndim=1] sol = np.asarray(solution, order='c') 
#    PyArray_ENABLEFLAGS(sol, np.NPY_OWNDATA)
#    
#    cdef np.ndarray[np.float_t, ndim=1] a_inv_r = np.asarray(a_inv, order='c')
#    PyArray_ENABLEFLAGS(a_inv_r, np.NPY_OWNDATA)
    
#    cdef np.ndarray[np.int_t, ndim=1] ia_inv_r = np.asarray(ia_inv2, order='c')
#    PyArray_ENABLEFLAGS(ia_inv_r, np.NPY_OWNDATA)
#    
#    cdef np.ndarray[np.int_t, ndim=1] ja_inv_r = np.asarray(ja_inv, order='c')
#    PyArray_ENABLEFLAGS(ja_inv_r, np.NPY_OWNDATA)
    
    #return sol,a_inv_r 

    #return np.asarray(solution, order='c')
    if not comp_selective_inverse:
        if comp_determinant:
            return (np.asarray(solution, order='c'), np.float(log_determ) )
        else:
            return (np.asarray(solution, order='c'), )
    else:
        if comp_determinant:    
            return (np.asarray(solution, order='c') , np.float(log_determ), np.asarray(a_inv, order='c'))
        else:
            return (np.asarray(solution, order='c'), np.asarray(a_inv, order='c'))

