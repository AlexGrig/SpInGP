# -*- coding: utf-8 -*-
"""
Binding pardiso library
"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

from libc.stdio cimport printf, sscanf
#from libc.stdlib import getenv
#cimport libc.math
from libc.stdlib cimport malloc, free 

#import numpy as np
#cimport numpy as np
#/* PARDISO prototype. */
name = "pardiso_bind"

cdef:
    extern void pardisoinit (void   *, int    *,   int *, int *, double *, int *);
    extern void pardiso     (void   *, int    *,   int *, int *,    int *, int *,
                             double *, int    *,    int *, int *,   int *, int *,
                             int *, double *, double *, int *, double *);
    extern void pardiso_chkmatrix  (int *, int *, double *, int *, int *, int *);
    extern void pardiso_chkvec     (int *, int *, double *, int *);
    extern void pardiso_printstats (int *, int *, double *, int *, int *, int *, double *, int *);

#p_ia = np.ascontiguousarray([ 0, 4, 7, 9, 11, 12, 15, 17, 20 ], dtype = np.int)
#cdef int* ia = &p_ia.data[0]

#p_ia = [ 0, 4, 7, 9, 11, 12, 15, 17, 20 ]
cdef int* ia = <int*> malloc(sizeof(int) * 9) 
ia[0] = 0; ia[1] = 4; ia[2] = 7; ia[3] = 9; ia[4] = 11; ia[5] = 12; ia[6] = 15;
ia[7] = 17; ia[8] = 20;          

#p_ja = [ 0, 2,  5, 6, 1, 2, 4, 2, 7, 3, 6, 1, 2, 5, 7, 1, 6, 2, 6, 7 ];
cdef int* ja = <int*> malloc(sizeof(int) * 20 ) 
ja[0] = 0; ja[1] = 2; ja[2] = 5; ja[3] = 6; ja[4] = 1; ja[5] = 2; ja[6] = 4; 
ja[7] = 2; ja[8] = 7; ja[9] = 3; ja[10] = 6; ja[11] = 1; ja[12] = 2; ja[13] = 5;
ja[14] = 7; ja[15] = 1; ja[16] = 6; ja[17] = 2; ja[18] = 6; ja[19] = 7;

#p_a = [ 7.0, 1.0, 2.0, 7.0, -4.0, 8.0, 2.0, 1.0, 5.0, 7.0, 9.0, -4.0, 7.0, 3.0, 
#     8.0, 1.0, 11.0, -3.0, 2.0, 5.0 ];
cdef double* a = <double*> malloc(sizeof(double) * 20 )
a[0] = 7.0; a[1] = 1.0; a[2] = 2.0; a[3] = 7.0; a[4] = -4.0; a[5] = 8.0; a[6] = 2.0; 
a[7] = 1.0; a[8] = 5.0; a[9] = 7.0; a[10] = 9.0; a[11] = -4.0; a[12] = 7.0; 
a[13] = 3.0; a[14] = 8.0; a[15] = 1.0; a[16] = 11.0; a[17] = -3.0; a[18] = 2.0; 
a[19] = 5.0;  

cdef:
     #/* Matrix data. */
    int    n = 8;
#    int    *ia #[ 9];
#    ia = [ 0, 4, 7, 9, 11, 12, 15, 17, 20 ];
#    int    *ja #[20];
#    ja = [ 0,    2,       5, 6,
#                         1, 2,    4,
#                            2,             7,
#                               3,       6,
#                         1,
#                            2,       5,    7,
#                         1,             6,
#                            2,          6, 7 ];


#    double  a[20];
#    a = [ 7.0,      1.0,   2.0, 7.0,
#                          -4.0, 8.0,      2.0,
#                                1.0,                     5.0,
#                                     7.0,           9.0,
#                          -4.0,
#                                7.0,           3.0,      8.0,
#                           1.0,                    11.0,
#                               -3.0,                2.0, 5.0 ];

    int      nnz = ia[n];
    int      mtype = 11;        #/* Real unsymmetric matrix */

    #/* RHS and solution vectors. */
    double   b[8]; 
    double x[8]; 
    double diag[8];
    int      nrhs = 1;          #/* Number of right hand sides. */

    #/* Internal solver memory pointer pt,                  */
    #/* 32-bit: int pt[64]; 64-bit: long int pt[64]         */
    #/* or void *pt[64] should be OK on both architectures  */
    void    *pt[64];

    #/* Pardiso control parameters. */
    int      iparm[64];
    double   dparm[64];
    int      solver;
    int      maxfct, mnum, phase, error, msglvl;

    #/* Number of processors. */
    int      num_procs=4;

    #/* Auxiliary variables. */
    char    *var;
    int      i, k;

    double   ddum;              #/* Double dummy */
    int      idum;              #/* Integer dummy. */

#/* -------------------------------------------------------------------- */
#/* ..  Setup Pardiso control parameters and initialize the solvers      */
#/*     internal adress pointers. This is only necessary for the FIRST   */
#/*     call of the PARDISO solver.                                      */
#/* ---------------------------------------------------------------------*/

error = 0;
solver = 0; #/* use sparse direct solver */
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


#/* Numbers of processors, value of OMP_NUM_THREADS */
#var = getenv("OMP_NUM_THREADS");
#if(var != NULL):
#    sscanf( var, "%d", &num_procs );
#else:
#    printf("Set environment OMP_NUM_THREADS to 1");
#    exit(1);

iparm[2]  = num_procs; # Alex: Manually set this, see comment above

iparm[10] = 0; #/* no scaling  */
iparm[12] = 0; #/* no matching */

maxfct = 1;         #/* Maximum number of numerical factorizations.  */
mnum   = 1;         #/* Which factorization to use. */

msglvl = 1;         #/* Print statistical information  */
error  = 0;         #/* Initialize error flag */


#/* -------------------------------------------------------------------- */
#/* ..  Convert matrix from 0-based C-notation to Fortran 1-based        */
#/*     notation.                                                        */
#/* -------------------------------------------------------------------- */

#for (i = 0; i < n+1; i++) {
#    ia[i] += 1;
#}
#for (i = 0; i < nnz; i++) {
#    ja[i] += 1;
#}
#
##/* Set right hand side to i. */
#for (i = 0; i < n; i++) {
#    b[i] = i;
#}

for i in range(n+1):
    ia[i] += 1;

for i in range(nnz):
    ja[i] += 1;

for i in range(n):
    b[i] = i;

#/* -------------------------------------------------------------------- */
#/*  .. pardiso_chk_matrix(...)                                          */
#/*     Checks the consistency of the given matrix.                      */
#/*     Use this functionality only for debugging purposes               */
#/* -------------------------------------------------------------------- */

pardiso_chkmatrix(&mtype, &n, a, ia, ja, &error);
if (error != 0):
    printf("\nERROR in consistency of matrix: %d", error);
    exit(1);


#/* -------------------------------------------------------------------- */
#/* ..  pardiso_chkvec(...)                                              */
#/*     Checks the given vectors for infinite and NaN values             */
#/*     Input parameters (see PARDISO user manual for a description):    */
#/*     Use this functionality only for debugging purposes               */
#/* -------------------------------------------------------------------- */

pardiso_chkvec(&n, &nrhs, b, &error);
if (error != 0):
    printf("\nERROR  in right hand side: %d", error);
    exit(1);

#/* -------------------------------------------------------------------- */
#/* .. pardiso_printstats(...)                                           */
#/*    prints information on the matrix to STDOUT.                       */
#/*    Use this functionality only for debugging purposes                */
#/* -------------------------------------------------------------------- */

pardiso_printstats(&mtype, &n, a, ia, ja, &nrhs, b, &error);
if (error != 0):
    printf("\nERROR right hand side: %d", error);
    exit(1);

#/* -------------------------------------------------------------------- */
#/* ..  Reordering and Symbolic Factorization.  This step also allocates */
#/*     all memory that is necessary for the factorization.              */
#/* ------------------------------------------------------------------- */
phase = 11;

pardiso(pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, &ddum, &ddum, &error,  dparm);

if (error != 0):
    printf("\nERROR during symbolic factorization: %d", error);
    exit(1);

printf("\nReordering completed ... ");
printf("\nNumber of nonzeros in factors  = %d", iparm[17]);
printf("\nNumber of factorization MFLOPS = %d", iparm[18]);

#/* -------------------------------------------------------------------- */
#/* ..  Numerical factorization.                                         */
#/* -------------------------------------------------------------------- */
phase = 22;

pardiso(pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, &ddum, &ddum, &error, dparm);

if (error != 0):
    printf("\nERROR during numerical factorization: %d", error);
    exit(2);

printf("\nFactorization completed ...\n ");

#/* -------------------------------------------------------------------- */
#/* ..  Back substitution and iterative refinement.                      */
#/* -------------------------------------------------------------------- */
phase = 33;

iparm[7] = 1;       #/* Max numbers of iterative refinement steps. */

pardiso(pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, b, x, &error,  dparm);

if (error != 0):
    printf("\nERROR during solution: %d", error);
    exit(3);

printf("\nSolve completed ... ");
printf("\nThe solution of the system is: ");

#for (i = 0; i < n; i++) {
#    printf("\n x [%d] = % f", i, x[i] );
#}

for i in range(n):
    printf("\n x [%d] = % f", i, x[i] );

printf ("\n");

#/* -------------------------------------------------------------------- */
#/* ..  Back substitution with tranposed matrix A^t x=b                  */
#/* -------------------------------------------------------------------- */

phase = 33;

iparm[7]  = 1;       #/* Max numbers of iterative refinement steps. */
iparm[11] = 1;       #/* Solving with transpose matrix. */

pardiso (pt, &maxfct, &mnum, &mtype, &phase,
         &n, a, ia, ja, &idum, &nrhs,
         iparm, &msglvl, b, x, &error,  dparm);

if (error != 0):
    printf("\nERROR during solution: %d", error);
    exit(3);

printf("\nSolve completed ... ");
printf("\nThe solution of the system is: ");
#for (i = 0; i < n; i++) {
#    printf("\n x [%d] = % f", i, x[i] );
#}

for i in range(n):
    printf("\n x [%d] = % f", i, x[i] );

printf ("\n");

#/* -------------------------------------------------------------------- */
#/* ... compute diagonal elements of the inverse.                        */
#/* -------------------------------------------------------------------- */

phase = 33;
iparm[11] = 0;       #/* Solving with nontranspose matrix. */
#/* solve for n right hand sides */
#for (k = 0; k < n; k++)
for k in range(n):
  #for (i = 0; i < n; i++) {
    for i in range(n):
        b[i] = 0;

  #/* Set k-th right hand side to one. */
    b[k] = 1;

    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
            &n, a, ia, ja, &idum, &nrhs,
           iparm, &msglvl, b, x, &error,  dparm);

    if (error != 0):
        printf("\nERROR during solution: %d", error);
        exit(3);
    #/* save diagonal element */
    diag[k] = x[k];


#/* -------------------------------------------------------------------- */
#/* ... Inverse factorization.                                           */
#/* -------------------------------------------------------------------- */
cdef int j;
if (solver == 0):

    printf("\nCompute Diagonal Elements of the inverse of A ... \n");
    phase = -22;
    iparm[35]  = 0; #/*  overwrite internal factor L */

    pardiso(pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs,
             iparm, &msglvl, b, x, &error,  dparm);

    #/* print diagonal elements */
    #for (k = 0; k < n; k++)
    for i in range(n):
        j = ia[k]-1;
        printf ("Diagonal element of A^{-1} = %32.24e =  %32.24e \n", a[j], diag[k]);


#/* -------------------------------------------------------------------- */
#/* ..  Convert matrix back to 0-based C-notation.                       */
#/* -------------------------------------------------------------------- */
#for (i = 0; i < n+1; i++) {
for i in range(n+1):
    ia[i] -= 1;

#for (i = 0; i < nnz; i++) {
for i in range(nnz):
    ja[i] -= 1;


#/* -------------------------------------------------------------------- */
#/* ..  Termination and release of memory.                               */
#/* -------------------------------------------------------------------- */
phase = -1;                 #/* Release internal memory. */

pardiso(pt, &maxfct, &mnum, &mtype, &phase,
         &n, &ddum, ia, ja, &idum, &nrhs,
         iparm, &msglvl, &ddum, &ddum, &error,  dparm);
