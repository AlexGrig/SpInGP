# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 16:55:47 2017

@author: alex
"""
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import solve_sym_pardiso

def build_sparse_matrix():
    """
    Build sparse matrix
    """
    N = 100
    A = np.zeros((N,N))
    np.fill_diagonal(A, np.arange(1,N+1))
    
    A[20,40] = 345;
    A[40,20] = 345.0;
    
    A[16,34] = 123;
    A[34,16] = 123;
    
    A = sparse.csr_matrix(A)
    
    np.random.seed(234)
    rhs = np.random.randn(100,1)
    return A,rhs

def solve(A,rhs):
    
    B = sparse.triu(A, format='csr')
    solution = solve_sym_pardiso.solve_linear_system(B.data, B.indptr, B.indices, np.squeeze(rhs) )
    #print(solution) 
    #print(A)
    
    sol_true = sla.spsolve(A, rhs)
    #print(rhs)
    #print(sol_true)
    print('Max abs diff: ', np.max(np.abs(solution - sol_true) ) )
    return solution

if __name__ == "__main__":
    A,rhs = build_sparse_matrix()
    sol = solve(A,rhs)
    