# -*- coding: utf-8 -*-
"""
Tests pardiso library solution speed.

@author: alex
"""
import os
import numpy as np
import scipy as sp
import gc
import sys

import time
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from TryCodes.pardiso import solve_sym_pardiso

from GPy.inference.latent_function_inference.ss_sparse_inference import btd_inference

from experiment_1 import generate_data, select_kernel

#save_file_prefix = '/u/85/agrigori/unix/Programming/python/Sparse GP/Experiemnts/Results'
save_file_prefix = '/home/alex/Programming/python/Sparse GP/Experiemnts/Results'

def build_matrix(N, kernel_no, p_largest_cond_num=1e14, p_regularization_type=2):
    """
    Builds the sparse matrix
    
    Inputs:
    -----------------------------
    N: int
        NUmber of data points
    kernel_no: object
        Kernel object
    ...
    
    Output:
    -----------------
    lil_sparse matrix
    """
    
    data_x, data_y = generate_data(N,p_x_lower_value=0.01, p_x_upper_value=2e4)
    kernel,_ = select_kernel(kernel_no)
    
    (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel.sde()
    
    block_size = F.shape[1]    
    
    diag_btd, low_diag_btd, log_det_btd, _, _, _, Ki_sparse = btd_inference.build_matrices(data_x, data_y, 
               F, L, Qc, P_inf, P0, H, p_largest_cond_num=p_largest_cond_num, p_regularization_type=p_regularization_type, 
               compute_derivatives=False, also_sparse=True)  
    
    #import pdb; pdb.set_trace()
    # Decreasing of non-diag Ki_diag elements
    #print(Diag, Low_diag)
    return sparse.csr_matrix(Ki_sparse), block_size

def solve_pardiso(A,rhs, supernode_size = -1, comp_determinant=True, inv_comp=True):
    """
    Solve linear symmetric system. Solution time is returned as well.
    
    Input:
    ---------------------------
    A: matrix 
        In any sparse or dence format
    rhs: vector
        Vector of right hand side.
        
    comp_determinant: bool
        Whether to compute a determinant
        
    inv_comp: bool
        Whether to compute the selective inverse. In this inverse only elements
        which are non-zero in the original matrix are returned.
        
    Return:
    ----------------------------
    t1, solution, log_determ, A_inv
    
    """
    
    #A_inv_true = sp.sparse.linalg.spsolve(A, np.eye( A.shape[0]))
    
    B = sparse.triu(A, format='csr')
    #B=A
    log_determ=None; A_inv = None
    
    #import pdb; pdb.set_trace()
    t1 = time.time()
    res = solve_sym_pardiso.solve_linear_system(B.data, B.indptr+1, B.indices+1, np.squeeze(rhs), supernode_size, comp_determinant, inv_comp ) 
    # +1 is done to transform to fortran # in python for paralellization.
    t1 = time.time() - t1
    
    if inv_comp:
        if comp_determinant:
            solution, log_determ, A_inv = res
        else:
            solution, A_inv = res
        A_inv = sparse.csr_matrix( (A_inv, B.indices, B.indptr) )
        A_inv = A_inv + sparse.triu(A_inv, k=1,format='csr').T
    else:
        if comp_determinant:
            solution, log_determ = res
        else:
            solution = res
    
    return t1, solution, log_determ, A_inv

def test1_solve_pardiso(N, kernel_no=0, set_supernode = False, comp_sparse=True, comp_sparse_inverse = True, p_largest_cond_num=1e14, p_regularization_type=2):
    """
    Simple test to verify pardiso integration and to compare with sparse solver.
    
    Input:
    ---------------
    
    set_supernode: bool
        Whether to set sepurnode size equal to block size in PARDISO library.
    
    comp_sparse: bool
        Compare with Scipy sparse at all.
        
    comp_sparse_inverse: bool
        Whether to compute the inverse of the sparse matrix by regular scipy method.
        This can cause large dense matrices, andhence not recommended for large N.
    """
    
    Matr , block_size = build_matrix(N, kernel_no, p_largest_cond_num=1e14, p_regularization_type=2)
    print('Run pariso test 1. N={0}, block_size={1}'.format(N,block_size) )
    sparse_rows = sp.sparse.coo_matrix(Matr).row 
    sparse_cols = sp.sparse.coo_matrix(Matr).col

    rhs = np.random.randn(N*block_size,1)
     
    #import pdb; pdb.set_trace()
    
    t1, sol_pardiso, log_det_pardiso, A_inv_pardiso = solve_pardiso(Matr, rhs, 
            supernode_size=block_size if set_supernode else -1, comp_determinant=True, inv_comp=True)
    #res= solve_pardiso(Matr, rhs, comp_determinant=True, inv_comp=True)
    #import pdb; pdb.set_trace()
    
# Sparse solver ->
    t2 = None
    if comp_sparse:
        lu = sparse.linalg.splu(Matr)
        log_det_sparse = np.sum( np.log(lu.U.diagonal()) )
        if comp_sparse_inverse:
            t2 = time.time() # Measure only the most time consuming operation.
            inv_sparse = sparse.linalg.spsolve(Matr, np.eye(Matr.shape[0]) )
            t2 = time.time() - t2
            # inv_sparse is a dense matrix now. Need to convert it to sparse with
            # indices as in A_inv_pardiso.
            A_inv_sparse = sparse.csr_matrix( np.zeros( Matr.shape, dtype=np.float) )
            A_inv_sparse[sparse_rows, sparse_cols] = inv_sparse[sparse_rows, sparse_cols]
            
            diff_inv_sparse = np.abs( A_inv_sparse - A_inv_pardiso )
            #diff_inv_sparse = diff_inv_sparse.todense()
            diff_max_inv_sparse = np.max(diff_inv_sparse)
        else:
            diff_max_inv_sparse = None
            
        sol_sparse = sparse.linalg.spsolve(Matr, rhs)
        
        diff_max_sol_sparse = np.max(np.abs(sol_sparse - sol_pardiso))
        diff_max_det_sparse = np.abs(log_det_sparse - log_det_pardiso)
    else:
        diff_max_inv_sparse = None; diff_max_det_sparse = None; diff_max_sol_sparse = None;
        log_det_pardiso = None; log_det_sparse = None;
# Sparse solver <-
    
    
    print('Max inverse matrix difference is:  {0}'.format(diff_max_inv_sparse) )
    print('Max log_det difference is:  {0}'.format(diff_max_det_sparse), '  ',  log_det_pardiso, log_det_sparse )
    print('Max solution difference is:  {0}'.format(diff_max_sol_sparse) )
    
    return block_size, t1, t2, diff_max_inv_sparse, diff_max_sol_sparse, diff_max_det_sparse
    
def ex3_a(file_name, N, kernel_no, repetition_no, set_supernode = False, comp_sparse = True, comp_sparse_inverse=True, p_largest_cond_num=1e14, p_regularization_type=2):
    """
    This function runs the inversion experiment and prints the output.
    
    Inputs:
    -----------------------------
    file_name: string
        Save file name
    N: int
        NUmber of data points
    kernel_no: int
        Kernel object
    
    repetition_no: int
        Number of repetitions
    
    set_supernode: bool
        Whether to set sepurnode size equal to block size in PARDISO library.
    
    comp_sparse: bool
        Compare with Scipy sparse at all.
        
    comp_sparse_inverse: bool
        Whether to compute the inverse of the sparse matrix by regular scipy method.
        This can cause large dense matrices, andhence not recommended for large N.
    ...
    
    Output:
    -----------------
    print output
        
    """

    file_name = os.path.join(save_file_prefix, file_name)
    
    times_1 = []; times_2 = []; diff_max_inv_sparse_list = [];
    diff_max_sol_sparse_list = []; diff_max_det_sparse_list = [];
    for rr in range(repetition_no):
        gc.collect()
        block_size, t1, t2, diff_max_inv_sparse, diff_max_sol_sparse, diff_max_det_sparse = test1_solve_pardiso(N, 
            kernel_no, set_supernode, comp_sparse_inverse, p_largest_cond_num=1e14, p_regularization_type=2)
    
        times_1.append(t1); times_2.append(t2); diff_max_inv_sparse_list.append(diff_max_inv_sparse);
        diff_max_sol_sparse_list.append(diff_max_sol_sparse); diff_max_det_sparse_list.append(diff_max_det_sparse)
    
    num_threads = os.environ.get('OMP_NUM_THREADS')
    
    import scipy.io as io
    result_dict = {}

    result_dict['num_threads'] = num_threads if num_threads is not None else 0
    result_dict['block_size'] = block_size
    result_dict['N'] = N
    
    #import pdb; pdb.set_trace()
    result_dict['times_1'] = times_1
    if times_2[-1] is None:
        times_2 = np.zeros(len(times_2))
    result_dict['times_2'] = np.zeros(len(times_2))
    
    if diff_max_inv_sparse_list[-1] is None:
        diff_max_inv_sparse_list = np.zeros(len(diff_max_inv_sparse_list))     
    result_dict['diff_max_inv_sparse_list'] = diff_max_inv_sparse_list
    
    if diff_max_sol_sparse_list[-1] is None:
        diff_max_sol_sparse_list = np.zeros(len(diff_max_sol_sparse_list))    
    result_dict['diff_max_sol_sparse_list'] = diff_max_sol_sparse_list
    
    if diff_max_det_sparse_list[-1] is None:
        diff_max_det_sparse_list = np.zeros(len(diff_max_det_sparse_list))
    result_dict['diff_max_det_sparse_list'] = diff_max_det_sparse_list
    
    io.savemat(file_name, result_dict)        
    print('OMP_NUM_THREADS: ', num_threads if num_threads is not None else 'Not set')
    print('set_supernode: ', set_supernode)
    
    print('PARDISO:  {0} datapoints, {1} block_size:  '.format(N,block_size), times_1,'s' )
    print('Scipy:  {0} datapoints, {1} block_size:  '.format(N,block_size), times_2,'s' )
    
    print('diff_max_inv_sparse_list:  {0} datapoints, {1} block_size:  '.format(N,block_size), diff_max_inv_sparse_list)
    print('diff_max_sol_sparse_list:  {0} datapoints, {1} block_size:  '.format(N,block_size), diff_max_sol_sparse_list)
    print('diff_max_det_sparse_list:  {0} datapoints, {1} block_size:  '.format(N,block_size), diff_max_det_sparse_list)

def plot_ex3_a(file_name_1, plot_thread_scaling=False, threads=(1,2,3,4)):
    """
    Prints or plots the scaling of Pardiso solver.

    Inputs:
    -------------------------
    file_name: string
        File name to print information about
        
    plot_thread_scaling: string
        Whether to plot graph for several threads no.
    
    threads: tuple
        If plot_thread_scaling is true, it asignes which threads (and files)
        are plotted.
    
    """
    import matplotlib.pyplot as plt
    
    file_name = os.path.join(save_file_prefix, file_name_1)
    import scipy.io as io
    result_dict = io.loadmat(file_name)
        
    num_threads = result_dict['num_threads'][0]
    block_size = result_dict['block_size']
    N = result_dict['N']
    times = result_dict['times_1']
    
    print('\nFile: ', file_name_1)
    print('  Num_threads: ', int(num_threads), ',N: ', int(N), ',block_size: ', int(block_size), ',time: ', times)
    
    if plot_thread_scaling:
        
        #threads = (1,2,3,4)
        thread_time_pardiso = []
        for th in threads:
            file_name = file_name[:-1] + str(th)
            file_name_full = os.path.join(save_file_prefix, file_name)
            result_dict = io.loadmat(file_name_full)
            
            num_threads = result_dict['num_threads']
            block_size = result_dict['block_size']
            N = result_dict['N']
            times = result_dict['times_1']
            
            thread_time_pardiso.append(np.mean(times))
        
        plt.figure(2)
        title_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 15}
        plt.title(r'Pardiso solve vs. Threads no. N={0}, block_size={1}'.format(int(N), int(block_size)), fontdict=title_font)    
        plt.plot(threads, thread_time_pardiso, 'ro-', label='BTD Matrix',markersize=8)

        labels_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 20}
        plt.xlabel('Threads', fontdict=labels_font)
        plt.ylabel('Time (Seconds)', fontdict=labels_font)
        plt.legend(loc=2)
        
        plt.yticks(fontsize=20)
        plt.xticks((1,2,3,4),('1','2','3','4'),fontsize=20)
        plt.show()
        
if __name__ == "__main__":
    
    if len(sys.argv) > 1: # file name is in command line
        file_name = sys.argv[1]
    else:
        file_name = 'tmp'
    
    #ex3_a(file_name, 20000, 5, 3, p_largest_cond_num=1e14, p_regularization_type=2)
    plot_ex3_a('pardiso_N_5e4_bs_16_sn_false_th4', plot_thread_scaling=True, threads=(1,2,4))
    
    #test1_solve_pardiso(20, kernel_no=0, p_largest_cond_num=1e14, p_regularization_type=2)
    #ex3_a(file_name, 50000, 4, 3, set_supernode =False, comp_sparse = False, comp_sparse_inverse=False, p_largest_cond_num=1e14, p_regularization_type=2)
