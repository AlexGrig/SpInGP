# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 17:49:10 2016

@author: alex
"""
import numpy as np
import time
from collections import Iterable
import scipy as sp 

#import GPy.inference.latent_function_inference.ss_sparse_inference as ss
#import ss_sparse_inference as ss
from GPy.inference.latent_function_inference import ss_sparse_inference
ss = ss_sparse_inference

import GPy

def generate_data(n_points):
    """
    Input:
    -----------
    
    n_points: number of data points.
    """

    x_lower_value = 0.0
    x_upper_value = 200.0


    x_points = np.linspace(x_lower_value, x_upper_value, n_points)


    # The signal is a sum of 2 sinusoids + noise
    
    noise_variance = 0.5
    # 1-st sunusoid
    magnitude_1 = 2.0
    phase_1 = 0.5
    period_1 = 23
    
    # 1-st sunusoid
    magnitude_2 = 5.0
    phase_2 = 0.1
    period_2 = 5
    
    y_points = magnitude_1*np.sin( x_points* 2*np.pi/ period_1 + phase_1) + \
                magnitude_2*np.sin( x_points* 2*np.pi/ period_2 + phase_2)
                
                
    y_points = y_points + noise_variance * np.random.randn(n_points,)            
    
    x_points.shape = (n_points,1)    
    y_points.shape = (n_points,1)
    
    return x_points, y_points
 
def test_random_matr_slicing_speeds(block_size, block_num, rep_number):
    """


    """    
    import time
    constr_dic = {}
    reading_dic = {}
    
    # Constructing a matrix:
    Block = np.random.random( (block_size,block_size) )     
    A_size = block_size * block_num
    # lil
    
    formats = ('lil', 'dok')
    
    for ff in formats:
        init_matr = sp.sparse.random(A_size, A_size)
        
        matr = init_matr.asformat(ff)
        
        constr_dic[ff] = []
        
        for rr in range(0,rep_number):
                                    
            t1 = time.time()                    
            matr[0:block_size, 0:block_size] = Block
            for ii in range(0,block_num-1):
                low_ind_start = ii*block_size
                low_ind_end = low_ind_start + block_size
                
                high_ind_start = (ii+1)*block_size
                high_ind_end =  high_ind_start + block_size
                
                matr[high_ind_start:high_ind_end, high_ind_start:high_ind_end] = Block + Block.T
                matr[low_ind_start:low_ind_end, high_ind_start:high_ind_end] = Block
                matr[high_ind_start:high_ind_end, low_ind_start:low_ind_end] = Block.T
            constr_dic[ff].append(time.time() - t1)

        constr_dic[ff] = np.mean( constr_dic[ff] )

    # Reading a matrix:
    formats = ('lil', 'csc', 'dok')
    
    m1 = matr.copy()
    
    for ff in formats:
        matr = m1.copy().asformat(ff)
        
        reading_dic[ff] = []
        
        for rr in range(0,rep_number):
                                    
            t2 = time.time()                    
            tmp = matr[0:block_size, 0:block_size]
            for ii in range(0,block_num-1):
                low_ind_start = ii*block_size
                low_ind_end = low_ind_start + block_size
                
                high_ind_start = (ii+1)*block_size
                high_ind_end =  high_ind_start + block_size
                
                tmp1 = matr[high_ind_start:high_ind_end, high_ind_start:high_ind_end]
                tmp2 = matr[low_ind_start:low_ind_end, high_ind_start:high_ind_end]
                tmp3 = matr[high_ind_start:high_ind_end, low_ind_start:low_ind_end]
                
            reading_dic[ff].append(time.time() - t2)

        reading_dic[ff] = np.mean( reading_dic[ff] )
    
    # Test matrix conversions:
    m2 = m1.asformat('lil')
    
    reading_dic['lil_to_csc'] = []
    for rr in range(0,rep_number):
        t3 = time.time()  
        tmp1 = m2.tocsc()   
        reading_dic['lil_to_csc'].append(time.time() - t3)
    reading_dic['lil_to_csc'] = np.mean( reading_dic['lil_to_csc'] )
    
    
    m2 = m1.asformat('csc')
    
    reading_dic['csc_to_lil'] = []
    for rr in range(0,rep_number):
        t3 = time.time()  
        tmp1 = m2.tolil()   
        reading_dic['csc_to_lil'].append(time.time() - t3)
    reading_dic['csc_to_lil'] = np.mean( reading_dic['csc_to_lil'] )


    return constr_dic,reading_dic

def test_sparse_determinant_computation():
    """    
    
    """
  
    np.random.seed(234) # seed the random number generator
        
    n_points = 1000
    x_data, y_data = generate_data(n_points)

    variance = 0.5
    lengthscale = 3.0
    period = 1.0 # For periodic
    kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)  
    #kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
    #kernel1 = GPy.kern.sde_RBF(1,variance=variance, lengthscale=lengthscale)
    noise_var = 0.1
    (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
    block_size = F.shape[1]    

    grad_calc_params = {}
    grad_calc_params['dP_inf'] = dP_inft
    grad_calc_params['dF'] = dFt
    grad_calc_params['dQc'] = dQct

    (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
     matrix_blocks_derivatives) = ss.sparse_inverse_cov(x_data, 
            y_data, F, L, Qc, P_inf, P0, H, compute_derivatives=True,
                       grad_calc_params=grad_calc_params)
                       
    HtH = np.dot(H.T, H)
    
    _, determ_1, tridiag_inv_data = ss.deriv_determinant( n_points, block_size, HtH, noise_var, 
                             matrix_blocks, compute_derivatives=False, deriv_num=None, 
                             matrix_derivs=matrix_blocks_derivatives, compute_inv_main_diag=False)
                                 
    # Compute the true inversion
    Ki = Ait*Qi*Ait.T # Precision(inverse covariance) without noise
    Ki = 0.5*(Ki + Ki.T)
    KiN = Ki +  GtG /noise_var# Precision with a noise        
    
    sp.sparse.linalg.eigsh(KiN, )
    
    
    
    
def test_matrix_cook_book_f164():
    """

    """

    n_points = 100
    x_data, y_data = generate_data(n_points)

    variance = np.random.uniform(0.1, 1.0) # 0.5
    lengthscale = np.random.uniform(0.2,10) #3.0
    noise_var = 0.1 #np.random.uniform( 0.01, 1) # 0.1
    
    kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
    
    gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
    gp_mean, gp_var = gp_reg.predict(x_data, include_likelihood=False)            

    #K = gp_reg.posterior.covariance
    K = gp_reg.kern.K(x_data)
    #other_mean = np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, y_data ) )
    #other_var = np.diag( K - np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, K) ) )[:,None]
    K_diag = K
    K_result = np.dot( K, np.linalg.solve(K + np.eye(n_points)*noise_var, K) )
    var1 = K_diag - K_result
    
    var2 = np.eye(n_points)*noise_var - noise_var*np.linalg.solve(K + np.eye(n_points)*noise_var, np.eye(n_points)*noise_var)

    #import pdb; pdb.set_trace()
    print( np.max(np.abs(var1 - var2)) )

def test_sparse_gp_timings(n_points, repetitions_num, kernel_num = 0):
    """
    This function returns necessary time measurements for sparse GP marginal
    likelihood and its derivatives computations

    """
    run_sparse = True    
    run_gp = True
    
    print('Sparse GP test %i' % n_points)
    x_data, y_data = generate_data(n_points)
    
    
    variance = 0.5
    lengthscale = 3.0
    period = 1.0 # For periodic
    if (kernel_num == 0): # blocksize is 2
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)
    elif (kernel_num == 1): # blocksize is 3
        kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)    
        kernel2 = GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale)
    elif (kernel_num == 2): # blocksize is 6
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale)       
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale)*GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale)
    elif (kernel_num == 3): # blocksize is 10
        kernel1 = GPy.kern.sde_RBF(1,variance=variance, lengthscale=lengthscale)
        kernel2 = GPy.kern.RBF(1,variance=variance, lengthscale=lengthscale)        
    elif (kernel_num == 4): # blocksize is 14 
        kernel1 = GPy.kern.sde_StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale)    
        kernel2 = GPy.kern.StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale )
    elif (kernel_num == 5): # blocksize is 20
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale) * \
            GPy.kern.sde_RBF(1,variance=variance, lengthscale=lengthscale)
        kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale) * \
            GPy.kern.RBF(1,variance=variance, lengthscale=lengthscale) 
    elif (kernel_num == 6): # blocksize is 28
        kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale) * \
                GPy.kern.sde_StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale)
        kernel2 =  GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale) * \
                    GPy.kern.sde_StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale)
    elif (kernel_num == 7): # blocksize is 42
        kernel1 = GPy.kern.sde_Matern52(1,variance=variance, lengthscale=lengthscale) * \
                GPy.kern.sde_StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale)  
        kernel2 = GPy.kern.Matern52(1,variance=variance, lengthscale=lengthscale) * \
                GPy.kern.StdPeriodic(1, variance=variance, period=period, lengthscale= lengthscale )
    
    
    noise_var = 0.1
    (F,L,Qc,H,P_inf, P0, dFt,dQct,dP_inft, dP0t) = kernel1.sde()
    block_size = F.shape[1]    
    
    grad_calc_params = {}
    grad_calc_params['dP_inf'] = dP_inft
    grad_calc_params['dF'] = dFt
    grad_calc_params['dQc'] = dQct
    
    timings_list = []
    total_timings = {};total_timings[0] = []; total_timings[1] = []; total_timings[2] = [];
    # Do not change with repetiotions
    timings_descr = None
    sparsity_info = None
    sparsity_descr = None
    
    mll_diff = None
    d_mll_diff = None
    d_mll_diff_relative = None
    for rr in range(0, repetitions_num):
        if run_sparse:
            print('Sparse GP run:')
            t1 = time.time()
            (Ait, Qi, GtY, G, GtG, H, Ki_derivatives, Kip, matrix_blocks, 
             matrix_blocks_derivatives) = ss.sparse_inference.sparse_inverse_cov(x_data, 
                    y_data, F, L, Qc, P_inf, P0, H, compute_derivatives=True,
                               grad_calc_params=grad_calc_params)
            total_timings[0].append(time.time() - t1)
            
            t1 = time.time()
            res = ss.sparse_inference.marginal_ll( block_size, y_data, Ait, Qi, \
            GtY, G, GtG, H, noise_var, compute_derivatives=True, dKi_vector=Ki_derivatives, 
                Kip=Kip, matrix_blocks= matrix_blocks, 
                matrix_blocks_derivatives = matrix_blocks_derivatives)
            total_timings[1].append(time.time() - t1)        
            
            marginal_ll = res[0]; d_marginal_ll = res[1]
            
            timings_list.append( res[2])
            timings_descr = res[3]
            sparsity_info = res[4]
            sparsity_descr = res[5]
        
        if run_gp:
            print('Regular GP run:')
            t1 = time.time()
            gp_reg = GPy.models.GPRegression(x_data,y_data,kernel2, noise_var=noise_var)
            
            mll_diff = marginal_ll - (-gp_reg.objective_function())
            
            reg_d_mll = -gp_reg.objective_function_gradients(); reg_d_mll.shape = (reg_d_mll.shape[0],1)
            d_mll_diff = d_marginal_ll - (reg_d_mll)
            d_mll_diff_relative = np.sum( np.abs(d_mll_diff) ) / np.sum( np.abs(reg_d_mll) ) 
            total_timings[2].append(time.time() - t1)
            
            del gp_reg
        
        print('Repetition no %i' % rr)

    def time_mean(i):
        """
        Means of timings over different iterations        
        
        Input:
        -------------
        
        i - index of interest
        """
        mean = 0
        for rr in range(0, repetitions_num):
            if isinstance( timings_list[rr][i], Iterable):
                mean += np.array( timings_list[rr][i] ).mean()
            else:
                mean += timings_list[rr][i]
        mean = mean / repetitions_num
        return mean

    def time_std(i):
        """
        Stds of timings over different iterations        
        
        Input:
        -------------
        
        i - index of interest
        """
        
        mean = time_mean(i)
        std = 0
        num = 0
        for rr in range(0, repetitions_num):
            if isinstance( timings_list[rr][i], Iterable):
                for val in timings_list[rr][i]:
                    std += (val - mean)**2
                    num += 1
            else:
                std += (timings_list[rr][i] - mean)**2
                num += 1
        std = np.sqrt( std / (num-1) )
        return std
    
    total_timings[0] = np.mean( total_timings[0] )
    total_timings[1] = np.mean( total_timings[1] )
    total_timings[2] = np.mean( total_timings[2] )
    return block_size, mll_diff , d_mll_diff, d_mll_diff_relative, timings_list, timings_descr, total_timings, sparsity_info, \
        sparsity_descr, time_mean, time_std

def scaling_measurement(result_file_name, each_size_rep_num, data_sizes=None):
    """
    Test scaling of the algorithm.
    
    Input:
    ------------------------
    result_file_name: text
    
    each_size_rep_num: int
        How much time each size computation is repeated.    
    
    """
    if data_sizes is None:
        data_sizes = (100,500,1000,3000,5000,7000,9000,10000)
    #data_sizes = (100,500)
    
    total_time_sparse = []
    total_time_reg_gp = []
    mll_diff_list = []
    d_mll_diff_list = []
    for ds in data_sizes:
        block_size, mll_diff , d_mll_diff, d_mll_diff_relative, timings_list, timings_descr, total_timings, sparsity_info, \
        sparsity_descr, time_mean, time_std = test_sparse_gp_timings(ds, each_size_rep_num)
        
        total_time_sparse.append( total_timings[0] + total_timings[1] )
        total_time_reg_gp.append(total_timings[2])
        mll_diff_list.append( np.abs(mll_diff) if mll_diff is not None else np.nan )
        d_mll_diff_list.append( np.sum(np.abs(d_mll_diff)) if d_mll_diff is not None else np.nan)
    
    
    import scipy.io as io
    result_dict = {}
    
    result_dict['data_sizes'] = data_sizes
    result_dict['total_time_sparse'] =  total_time_sparse   
    result_dict['total_time_reg_gp'] = total_time_reg_gp
    result_dict['mll_diff_list'] = mll_diff_list
    result_dict['d_mll_diff_list'] = d_mll_diff_list
    
    io.savemat(result_file_name, result_dict)
    
def plot_scaling_measurements(file_name):
    
    import matplotlib.pyplot as plt
    import scipy.io as io
    
    result_dict = io.loadmat(file_name)
    
    data_sizes = result_dict['data_sizes'].squeeze()
    total_time_sparse = result_dict['total_time_sparse'].squeeze()
    total_time_reg_gp = result_dict['total_time_reg_gp'].squeeze()
    mll_diff_list = result_dict['mll_diff_list'].squeeze()
    d_mll_diff_list = result_dict['d_mll_diff_list'].squeeze()
     
    
    plt.figure(1)
    plt.title('Running Times Comparison' )    
    plt.plot( data_sizes, total_time_reg_gp, 'bo-', label='reg gp')
    plt.plot( data_sizes, total_time_sparse, 'ro-', label='sparse gp')        
    plt.xlabel('Sample Length')
    plt.ylabel('Time (Seconds)')
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.title('Mll and d_Mll discrepancies')    
    plt.plot( data_sizes, mll_diff_list, 'bo-', label='mll (abs)')
    plt.plot( data_sizes, d_mll_diff_list, 'ro-', label='d_mll (L1)')        
    plt.xlabel('Sample Length')
    plt.ylabel('Abs or L1 norm')
    plt.legend()
    plt.show()


def experiment_1():
    """
    Test scaling of sparse GP with comparison to regular GP.
    After code optimization.    
    
    """
    #!!! Set run_gp = True in test_sparse_gp_timings    
    
    
    data_sizes = (100,500,1000,3000,5000,7000,9000,10000)
    # After 10000 memory swaping becomes obvioous
    
    scaling_measurement('first_result',5, data_sizes)
    plot_scaling_measurements('first_result')


def experiment_2():
    """
    Test scaling of Matern32 (block_size 2) of sparse GP.
    """
    #!!! Set run_gp = False in test_sparse_gp_timings
    
    data_sizes = (1000,5000,10000,20000,30000,40000,50000)
    # After 10000 memory swaping becomes obvioous
    
    scaling_measurement('second_result',5, data_sizes)
    plot_scaling_measurements('second_result')

def experiment_3(plot_only=False):
    """
    Test scaling of sparce GP wrt block sizes (different kernels)
    """   
    #!!! Set run_gp = True in test_sparse_gp_timings. We want to compare accuracy with
    # regular GP.
    result_file_name = 'experiment3'
    
    kernel_nums = (0,1,2,3,4,5,6,7)
    #kernel_nums = (0,1,)
    
    if not plot_only:
        block_size_list = []
        mll_diff_list = []
        d_mll_diff_list = []
        d_mll_diff_relative_list = []
        total_timings_list = []
        for kk in kernel_nums:
            block_size, mll_diff , d_mll_diff, d_mll_diff_relative, timings_list, timings_descr, total_timings, sparsity_info, \
            sparsity_descr, time_mean, time_std = test_sparse_gp_timings(1000, 3, kernel_num = kk)
            
            block_size_list.append( block_size )
            mll_diff_list.append( mll_diff[0,0] )
            d_mll_diff_list.append( d_mll_diff )
            d_mll_diff_relative_list.append( d_mll_diff_relative )
            total_timings_list.append(total_timings )
            
#        import scipy.io as io
        result_dict = {}
        
        result_dict['block_size_list'] = block_size_list
        result_dict['mll_diff_list'] =  mll_diff_list   
        result_dict['d_mll_diff_list'] = d_mll_diff_list
        result_dict['total_timings_list'] = total_timings_list
#        
#        io.savemat(result_file_name, result_dict)
#    
#    result_dict = io.loadmat(result_file_name)    
#    
#    block_size_list = result_dict['block_size_list']
#    mll_diff_list = result_dict['mll_diff_list'] 
#    d_mll_diff_list = result_dict['d_mll_diff_list']
#    total_timings_list = result_dict['total_timings_list']
    
    import matplotlib.pyplot as plt
    
    cov_matrs_time_iter = [total_timings_list[i][0] for i in range(0, len(total_timings_list))]
    d_mll_time_iter = [total_timings_list[i][1] for i in range(0, len(total_timings_list))]
    reg_gp_time_iter = [total_timings_list[i][2] for i in range(0, len(total_timings_list))]
    
    plt.figure(1)
    plt.title('Run Times (Block_sizes). Samples 1000' )    
    plt.plot( block_size_list, cov_matrs_time_iter, 'mo-', label='sparce gp (cov_matrs)')
    plt.plot( block_size_list, d_mll_time_iter, 'ro-', label='sparse gp (d_mll)')
    plt.plot( block_size_list, reg_gp_time_iter, 'bo-', label='reg gp')        
    plt.xlabel('Block_sizes')
    plt.ylabel('Time (Seconds)')
    plt.legend()
    plt.show()
    
    d_mll_l1_relative_iter = [d_mll_diff_relative_list[i] for i in range(0, len(total_timings_list))]
    d_mll_l1_iter = [np.sum(np.abs(d_mll_diff_list[i])) for i in range(0, len(d_mll_diff_list))]
    
    plt.figure(2)
    plt.title('Mll discrepancies. Samples 1000')    
    plt.plot( block_size_list, np.abs(mll_diff_list), 'bo-', label='mll (abs)')       
    plt.xlabel('Block_sizes')
    plt.ylabel('Abs')
    plt.legend()
    plt.show()
    
    plt.figure(3)
    plt.title('d_Mll discrepancies. Samples 1000')
    plt.plot( block_size_list, d_mll_l1_iter, 'ro-', label='d_mll (L1)')        
    plt.xlabel('Block_sizes')
    plt.ylabel('L1 norm')
    plt.legend()
    plt.show()
    
    plt.figure(4)
    plt.title('d_Mll relative discrepancies. Samples 1000')
    plt.plot( block_size_list, d_mll_l1_relative_iter, 'ro-', label='d_mll (L1)')        
    plt.xlabel('Block_sizes')
    plt.ylabel('L1 norm / L1 norm')
    plt.legend()
    plt.show()
    
    return result_dict
        
        
        
        
if  __name__ == '__main__':
    #test_sparse_inverse_cov_simple()
    #pass
    #V31021362
    
    #experiment_1()
    #experiment_2()
    #experiment_3()
    
    #0, 1, 2, 4 - in tests. The rest - large gradient difference
    #block_size, mll_diff , d_mll_diff, d_mll_diff_relative, timings_list, timings_descr, total_timings, sparsity_info, \
    #    sparsity_descr, time_mean, time_std = test_sparse_gp_timings(100, 1, kernel_num = 6) # 3-, 4-, 5-, 6-, 7-
    test_matrix_cook_book_f164()
    
    #scaling_measurement('second_result')
    #plot_scaling_measurements('second_result')
    
    #constr_times, read_times = test_random_matr_slicing_speeds(10, 1000, 5)