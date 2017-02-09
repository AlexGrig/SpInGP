# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 18:35:12 2016

@author: alex
"""
import numpy as np
import scipy as sp
import scipy.linalg as la
from scipy import sparse

import sksparse.cholmod as cholmod


def test_sksparse_cholesky_update():
    """
    General test
    """
    print("Regular Cholesky:")
    A = np.array( ((11,2,3,),(2,5,6,),(3,6,9.0))  )
    print(A)
    L = la.cholesky(A, lower=True)
    print(L)
    print(np.dot(L,L.T))

    vector = np.array( ((0.1,),(3,),(0.3,)) )
    A_upd = A + np.dot(vector, vector.T)
    L_upd = la.cholesky(A_upd, lower=True)
    print('Updated Regular Cholesky:')
    print(L_upd)
    
    
    print('\n')
    print('Sparse Cholesky:')
    A_csc = sparse.csc_matrix(A)
    A_factor = cholmod.cholesky(A_csc,)
    print(A_factor.L().toarray())
    
    print('P matrix:')
    print(A_factor.P())
    
    A_factor.update_inplace(sparse.csc_matrix(vector))
    print('Updated Sparse Cholesky')
    L_sparse_update = A_factor.L().toarray()
    
    print(L_sparse_update) 
    print( L_upd - L_sparse_update )    


def test_triangular_solver():
    """
    Test triangular system solver.
    """
    
    print("Regular Cholesky:")
    A = np.array( ((11,2,3,),(2,5,6,),(3,6,9.0))  )
    b = np.array(((3.0,),(2.0,),(1.0,)) )
    print(A)
    L = la.cholesky(A, lower=True)
    print(L)
    x1 = la.solve_triangular(L, b, lower=True)
    print(x1)
    
    print('\n')
    print('Sparse Cholesky:')
    A_csc = sparse.csc_matrix(A)
    A_factor = cholmod.cholesky(A_csc,)
    print(A_factor.L().toarray())
    print('P matrix:')
    print(A_factor.P())
    print('D matrix:')
    D = A_factor.D()
    print(D)
    
    x2 = A_factor.solve_L(b)
    print('x2:')
    print(x2)    
    
    print('x3:')
    x3 =  np.dot( np.diag(1.0/np.sqrt(D)), x2)
    print(x3)
    
def test_permutaion_solving():
    """
    This function is similar to the previous one, except that
    
    """
    
    M = np.array([[  148.87,    42.61,  -147.78,    43.47,     0.  ,     0.  ],
       [   42.61,    20.46,   -43.47,     8.63,     0.  ,     0.  ],
       [ -147.78,   -43.47,  4055.39,   346.41, -3907.5 ,   389.88],
       [   43.47,     8.63,   346.41,    69.47,  -389.88,    25.97],
       [    0.  ,     0.  , -3907.5 ,  -389.88,  3908.53,  -389.02],
       [    0.  ,     0.  ,   389.88,    25.97,  -389.02,    55.01]])
    
    B = np.array( ((0.0, 0 ,3.0, 0, 0, 1),) ).T
    
    M_csc = sparse.csc_matrix(M)
    M_factor = cholmod.cholesky(M_csc,)
    
    Bp = M_factor.apply_P(B)
    s1 = M_factor.solve_LDLt( Bp )   
    s2 =  M_factor.apply_Pt(s1)    
    print('Perm solution:')    
    print(s2)    
    
    s_reg = np.linalg.solve(M,B)
    print('Regular solution:')    
    print(s_reg) 
    
    print(s2 - s_reg)

def test_nonpositive_def():
    """
    Test Cholesky of non positive definite matrices.
    """    
    
    import scipy.io as io
    
    res = io.loadmat('dk.mat')
    dk = res['dk'] #+ 1000000* sparse.eye(6, format='csc' )  
    
    dk_factor = cholmod.cholesky(dk,)
    L = dk_factor.L()    
    L1 = dk_factor.apply_P( dk_factor.L() )
    L2 = dk_factor.apply_Pt( dk_factor.L() )
    
    print( (L*L.T-dk).sum() )
    print( (L1*L1.T-dk).sum() )    
    print( (L2*L2.T-dk).sum() )
    
    # If matrix is positive definite this approach works if not then it does not.    
    dk = res['dk']
    dk_factor = cholmod.cholesky(dk,)
    (Ld,Dd) = dk_factor.L_D()
    Ld = Ld.tocsc(); Dd = Dd.tocsc();
    
    L1d = dk_factor.apply_P( Ld )
    L2d = dk_factor.apply_Pt( Ld ) 
    print( (Ld*Dd*Ld.T-dk).sum() )
    print( (L1d*Dd*L1d.T-dk).sum() )    
    print( (L2d*Dd*L2d.T-dk).sum() )
    
    # Cholesky for positive definite ->
    A=sparse.rand(6,6,density=1, format='csc') + sparse.eye(6, format='csc' )
    A = A + A.T
    
    A_factor = cholmod.cholesky(A,)
    AL = A_factor.L()
    AL1 = A_factor.apply_P( A_factor.L() )
    AL2 = A_factor.apply_Pt( A_factor.L() )
    
    
    print( (AL*AL.T-A).sum() )
    print( (AL1*AL1.T-A).sum() )    
    print( (AL2*AL2.T-A).sum() )
    # Cholesky for positive definite <-    
    
if __name__ == '__main__':
    #test_sksparse_cholesky_update()
    #test_triangular_solver()
    #test_permutaion_solving()
    test_nonpositive_def()
        
    