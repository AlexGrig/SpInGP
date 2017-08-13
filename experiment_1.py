# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 18:14:00 2017
 
@author: alex
"""
import numpy as np
import scipy as sp
import time
import os
import sys # for receiving command line arguments
import gc # garbage collector

import GPy
import GPy.models.ss_sparse_model as ss_sparse_model # do we need this?


#save_file_prefix = '/u/85/agrigori/unix/Programming/python/Sparse GP/Experiemnts/Results'
save_file_prefix = '/home/alex/Programming/python/Sparse GP/Experiemnts/Results'

def generate_data(n_points, p_x_lower_value=None, p_x_upper_value=None):
    """
    The function generates data. THe data is the sum
    of two sinusoids the the parameters given in the code     
    
    
    Input:
    --------------------
    
    n_points: int
        Number of data points.
        
        
    Output:
    -------------------
    
    x_points, y_points: arrays
        Generated data.
    
    """
    if p_x_lower_value is None:
        x_lower_value = 0.0
    else:
        x_lower_value = p_x_lower_value
    
    if p_x_upper_value is None:
        x_upper_value = 200.0
    else:
        x_upper_value = p_x_upper_value

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

def select_kernel(kernel_num = 0):
    """
    Choose kernel by its integer number
    
    Input:
    ----------------------
    kernel: int
        Kernel index
        
    Output:
        2 corresponding kernel objects: first sde, second regular.
    """
    
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
    elif (kernel_num == 4): # blocksize is 16 
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
    
    return kernel1, kernel2
    
def model_time_measurement(n_points, kernel_num = 0, repetitions_num = 3, run_sparse = True, run_ss = True, run_gp = True):
    """
    This function builds a model (in GPy sense) and measure the time it took.
    
    Input:
    ------------------------------
    
    n_points: int
        Number of data points in the data
    kenrel_num: int
        Which kernel to use. For State-space and sparse model the kernel defines
        the block size in the inverse kernel BTD matrix.
    repetitions_num: int
        How many times time measurement is performed
    run_sparse: bool
        Whether to run the time measurement for SpiIn-GP
    run_ss: bool
        Whether to run the time measurement for State-space model
    run_gp: bool
        Whether to run the time measurement for State-space model
        
    Output:
    ----------------------
    Mean running time
    
    """
    
    print('Time measurement test: %i data points, run_sparse %i, run_ss %i , run_gp %i' % (n_points, run_sparse, run_ss, run_gp))
    x_data, y_data = generate_data(n_points, p_x_lower_value=0.0, p_x_upper_value=20000.0)
    
    kernel1, kernel2 = select_kernel(kernel_num)
    noise_var = 0.1

    sparse_run_time=[]
    ss_run_time=[]
    gp_run_time=[]
    gc.collect()    
    for rep_no in range(repetitions_num):
        if run_sparse:
            #print('Sparse GP run:')
            kern = kernel1.copy()
            
            t1 = time.time()
            sparse_gp = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel=kern, noise_var=noise_var, balance=False, 
                                                      largest_cond_num=1e+10, regularization_type=2)
            sparse_run_time.append(time.time() - t1)
            
            sparse_gp_marginal_ll = (-sparse_gp.objective_function())
            sparse_d_marginal_ll = -sparse_gp.objective_function_gradients(); sparse_d_marginal_ll.shape = (sparse_d_marginal_ll.shape[0],1)
            
            del sparse_gp, kern
            gc.collect()
        if run_ss:
            #print('SS run:')
            kern = kernel1.copy()
            
            t1 = time.time()
            ss_model = GPy.models.StateSpace( x_data, y_data, kernel=kern, noise_var=noise_var, balance=False, kalman_filter_type = 'svd')
            
            ss_run_time.append(time.time() - t1)
            
            #ss_marginal_ll = (-ss_model.objective_function())
            #ss_d_marginal_ll = -ss_model.objective_function_gradients(); ss_d_marginal_ll.shape = (ss_d_marginal_ll.shape[0],1)
            
            del ss_model, kern
            gc.collect()
        if run_gp:
            #print('Regular GP run:')
            kern = kernel2.copy()
            
            t1 = time.time()
            gp_reg = GPy.models.GPRegression(x_data,y_data, kern, noise_var=noise_var)
            gp_run_time.append(time.time() - t1)
            
            #gp_marginal_ll = (-gp_reg.objective_function())
            #gp_d_marginal_ll = -gp_reg.objective_function_gradients(); gp_d_marginal_ll.shape = (gp_d_marginal_ll.shape[0],1)
            
            del gp_reg, kern
            gc.collect()
    gc.collect()
    return sparse_run_time, ss_run_time, gp_run_time
    
def scaling_measurement(result_file_name, kernel_no, each_size_rep_num, data_sizes=None,
                        run_sparse = True, run_ss = True, run_gp = True):
    """
    Test scaling of the different models wrt data size.
    
    Input:
    ------------------------
    result_file_name: text
    
    kernel_no: int
        Number of the kernel
        
    each_size_rep_num: int
        How much time each size computation is repeated.    
    
    run_sparse, run_ss, run_gp: bools
        Which models are used.
    """

    result_file_name = os.path.join(save_file_prefix, result_file_name)
    #import pdb; pdb.set_trace()
    
    if data_sizes is None:
        data_sizes = (100,500,1000,3000,5000,7000,9000,10000)
        #data_sizes = (100,500,1000)
    
    total_time_sparse = []
    total_time_ss = []
    total_time_reg_gp = []
    
    for ds in data_sizes:
         sparse_run_time, ss_run_time, gp_run_time = \
         model_time_measurement(ds, kernel_num=kernel_no, repetitions_num=each_size_rep_num, 
                                run_sparse = run_sparse, run_ss = run_ss, run_gp = run_gp)
                 
         total_time_sparse.append( sparse_run_time )
         total_time_reg_gp.append( gp_run_time )
         total_time_ss.append( ss_run_time )
    
    import scipy.io as io
    result_dict = {}
    
    result_dict['data_sizes'] = data_sizes
    result_dict['total_time_sparse'] =  total_time_sparse   
    result_dict['total_time_reg_gp'] = total_time_reg_gp
    result_dict['total_time_ss'] = total_time_ss
    
    io.savemat(result_file_name, result_dict)
    
def plot_scaling_measurements(file_name, plot_sparse = True, plot_ss = True, plot_gp = True, 
                              plot_thread_scaling=False, thread_scaling_data_size=1000):
    """
    The function plots the result of scaling experiment.
    
    Inputs:
    -------------------------------
    file_name:
    
    plot_sparse, plot_ss, plot_gp: bool
        Whether to plot corresponding components
    
    plot_thread_scaling: bool
        If the same experiment has been performed for different thread number, then
        this parameter says whether to plot this thead graph.
        
    thread_scaling_data_size: int
        For which data size to plot the thread graph.
        
    """
    import matplotlib.pyplot as plt
    import scipy.io as io    
    
    file_name_full = os.path.join(save_file_prefix, file_name)
    
    result_dict = io.loadmat(file_name_full)
    
    data_sizes = result_dict['data_sizes'].squeeze(); print('Data sizes:', '\n', data_sizes)
    total_time_sparse = result_dict['total_time_sparse'].squeeze(); print('total_time_sparse:', '\n', total_time_sparse)
    print('Means:',  np.mean(total_time_sparse,axis=1))
    total_time_reg_gp = result_dict['total_time_reg_gp'].squeeze(); print('total_time_reg_gp:', '\n', total_time_reg_gp)
    print('Means:',  np.mean(total_time_reg_gp,axis=1))
    total_time_ss = result_dict['total_time_ss'].squeeze(); print('total_time_ss:', '\n', total_time_ss)
    print('Means:',  np.mean(total_time_ss,axis=1))
    
    plt.figure(1)
    title_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 22}
    plt.title(r'Likelihood Computation Running Times', fontdict=title_font)
    
    if plot_sparse: 
        plt.plot( data_sizes, np.mean(total_time_sparse,axis=1), label='SpIn GP', linestyle='-', marker= 'o', color='r', markersize=9, linewidth=2)
    if plot_gp:
        plt.plot( data_sizes, np.mean(total_time_reg_gp,axis=1), label='Regular GP', linestyle='--', marker= 's', color='b', markersize=9, linewidth=2)
    if plot_ss:
        plt.plot( data_sizes, np.mean(total_time_ss,axis=1), label='Kalman Filter (State-space model)', linestyle='-.', marker= '^', color='g', markersize=9, linewidth=2)
    
    labels_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 20}
    plt.xlabel('Sample Length', fontdict=labels_font)
    plt.ylabel('Time (Seconds)', fontdict=labels_font)
    plt.tight_layout() # for adjusting the bottom of x label
    plt.legend(loc=2)
    
    from matplotlib.ticker import AutoMinorLocator
    ax = plt.gca()
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    plt.tick_params(which='both', width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4)
        
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()
    
    if plot_thread_scaling:
        #search_corresponding index of thread_scaling_data_size
        data_size_index = (100,500,1000,3000,5000,7000,9000,10000).index(thread_scaling_data_size)
        
        threads = (1,2,3,4)
        thread_time_sparse = []
        thread_time_gp = []
        thread_time_ss = []
        for th in range(4):
            file_name = file_name[:-1] + str(th+1)
            file_name_full = os.path.join(save_file_prefix, file_name)
            result_dict = io.loadmat(file_name_full)
            
            total_time_sparse = result_dict['total_time_sparse'].squeeze()[data_size_index]
            total_time_reg_gp = result_dict['total_time_reg_gp'].squeeze()[data_size_index]
            total_time_ss = result_dict['total_time_ss'].squeeze()[data_size_index]
            
            thread_time_sparse.append(np.mean(total_time_sparse))
            thread_time_gp.append(np.mean(total_time_reg_gp))
            thread_time_ss.append(np.mean(total_time_ss) ) 
        
        plt.figure(2)
        plt.title(r'Likelihood wrt Number of Threads', fontdict=title_font)    
        
        if plot_sparse:
            plt.plot( threads, thread_time_sparse, label='SpIn GP', linestyle='-', marker= 'o', color='r', markersize=9, linewidth=2)
        if plot_gp:
            plt.plot( threads, thread_time_gp, label='Regular GP', linestyle='--', marker= 's', color='b', markersize=9, linewidth=2)
        if plot_ss:
            plt.plot( threads, thread_time_ss, label='Kalman Filter (State-space model)', linestyle='-.', marker= '^', color='g', markersize=9, linewidth=2)
    
        plt.xlabel('Threads', fontdict=labels_font)
        plt.ylabel('Time (Seconds)', fontdict=labels_font)
        plt.tight_layout() # for adjusting the bottom of x label
        plt.legend()
        
        ax = plt.gca()
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    
        plt.tick_params(which='both', width=1)
        plt.tick_params(which='major', length=7)
        plt.tick_params(which='minor', length=4)
    
        plt.yticks(fontsize=20)
        plt.xticks( (1,2,3,4),('1','2','3','4'), fontsize=20)
        plt.show()
          
def block_size_scaling(result_file_name, each_size_rep_num=1, data_size=1000, kernel_nos=None):
    """
    The function computes the scaling of Spin-GP and State-Space model wrt block sizes.
    
    Input:
    ------------------
    result_file_name: string
    
    each_size_rep_num: string
        number of repetitions
        
    data_size: int
        data size
        
    kernel_nos: list or array
        Which kernels to use
        
    Output:
    ---------------------
    Results in a file
    """

    result_file_name = os.path.join(save_file_prefix, result_file_name)
    
    block_sizes = (2,3,6,10,14,20,28,42) # for the kernels (0,1,2,3,4,5,6,7)
    if kernel_nos is None:
        kernel_nos = (0,1,2,3,4,5,6,7)
    
    total_time_sparse = []
    total_time_ss = []
    total_time_reg_gp = []
    
    for kr in kernel_nos:
        sparse_run_time, ss_run_time, gp_run_time = \
         model_time_measurement(data_size, kernel_num=kr, repetitions_num=each_size_rep_num, 
                                run_sparse = True, run_ss = True, run_gp = True)
        
        total_time_sparse.append( sparse_run_time )
        total_time_reg_gp.append( gp_run_time )
        total_time_ss.append( ss_run_time )
        
    import scipy.io as io
    result_dict = {}
    
    result_dict['data_size'] = data_size
    result_dict['block_sizes'] = np.array(block_sizes)[np.array(kernel_nos)]
    
    result_dict['kernel_nos'] = kernel_nos
    result_dict['total_time_sparse'] =  total_time_sparse   
    result_dict['total_time_reg_gp'] = total_time_reg_gp
    result_dict['total_time_ss'] = total_time_ss
    
    io.savemat(result_file_name, result_dict)

def plot_block_size_scaling(file_name):
    """
    The function plots the result of block scaling experiment
    """
    
    import matplotlib.pyplot as plt
    import scipy.io as io    
    
    
    file_name = os.path.join(save_file_prefix, file_name)
    result_dict = io.loadmat(file_name)
    
    data_size = float(result_dict['data_size'].squeeze()); print('Data size:', '\n', data_size)
    block_sizes = result_dict['block_sizes'].squeeze(); print('Block sizes:', '\n', block_sizes)
    kernel_nos = result_dict['kernel_nos'].squeeze(); print('Kernel nos:', '\n', kernel_nos)    
    
    total_time_sparse = result_dict['total_time_sparse'].squeeze(); print('total_time_sparse:', '\n', total_time_sparse)
    print('Means:',  np.mean(total_time_sparse,axis=1))
    total_time_reg_gp = result_dict['total_time_reg_gp'].squeeze(); print('total_time_reg_gp:', '\n', total_time_reg_gp)
    print('Means:',  np.mean(total_time_reg_gp,axis=1))
    total_time_ss = result_dict['total_time_ss'].squeeze(); print('total_time_ss:', '\n', total_time_ss)
    print('Means:',  np.mean(total_time_ss,axis=1))

    
    plt.figure(3)
    title_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 22}
    plt.title('Kernel block size influence', fontdict=title_font)
    
    plt.plot( block_sizes[:-1], np.mean(total_time_sparse,axis=1)[:-1], label='SpIn GP', linestyle='-', marker= 'o', color='r', markersize=9, linewidth=2)
    plt.plot( block_sizes[:-1], np.mean(total_time_reg_gp,axis=1)[:-1], label='Regular GP', linestyle='--', marker= 's', color='b', markersize=9, linewidth=2)
    plt.plot( block_sizes[:-1], np.mean(total_time_ss,axis=1)[:-1], label='Kalman Filter (State-space model)', linestyle='-.', marker= '^', color='g', markersize=9, linewidth=2)

    labels_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 20}
    plt.xlabel('Kernel block size', fontdict=labels_font)
    plt.ylabel('Time (Seconds)', fontdict=labels_font)
    plt.legend(loc=2)
    plt.tight_layout() # for adjusting the bottom of x label
    
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.show()
    
if __name__ == "__main__":
    # get the file name from command line arguments
    if len(sys.argv) > 1: # file name is in command line
        file_name = sys.argv[1]
    else:
        file_name = 'tmp' # if not teke file from here
        
    if len(sys.argv) > 2: # the second argument is the task. Task start from 1
        task = int(sys.argv[2])
    else:
        task = 0
        
    num_threads = os.environ.get('OMP_NUM_THREADS')
    print('OMP_NUM_THREADS: ', num_threads if num_threads is not None else 'Not set')
    
    if task==1:
        print('task 1: ', file_name)
        scaling_measurement(file_name, kernel_no=5, each_size_rep_num=3, data_sizes=None)
    elif task==2:
        print('task 2: ', file_name)
        block_size_scaling(file_name, each_size_rep_num=3, data_size=1000, kernel_nos=None)
    else:
        print('task else: ', file_name)
    
    # ================================================================================
    ## (run) ex1 sample small. Compare Kalman filter, SpInGP and GP 
    #scaling_measurement('scaling_measurement_small_th4', kernel_no=5, each_size_rep_num=3,  
    #                    run_sparse = True, run_ss = True, run_gp = True)
    ## (plot) ex1 sample small. Compare Kalman filter, SpInGP and GP 
    #plot_scaling_measurements('scaling_measurement_small_th4', 
    #                          plot_sparse = True, plot_ss = True, plot_gp = True,                             
    #                          plot_thread_scaling=False, thread_scaling_data_size=10000)
    
    
    
    # ================================================================================
    ## (run) ex1 sample large. Compare Kalman filter and SpInGP 
    #scaling_measurement('scaling_measurement_large_th4', kernel_no=5, each_size_rep_num=3, data_sizes=(1000,5000,10000,20000,30000,40000,50000), 
    #                    run_sparse = True, run_ss = True, run_gp = False)
    ## (plot) ex1 sample large. Compare Kalman filter and SpInGP 
    #plot_scaling_measurements('scaling_measurement_large_th4', 
    #                          plot_sparse = True, plot_ss = True, plot_gp = False,                             
    #                          plot_thread_scaling=False, thread_scaling_data_size=10000)
    
    # ================================================================================
    ## (run) Block size scaling
    #block_size_scaling('test_block_size_scaling', each_size_rep_num=3, data_size=1000, kernel_nos=(0,1,2,3,4,5,6))
    
    ## (plot) Block size scaling
    #plot_block_size_scaling('block_size_scaling_th4')
        
    
    # ================================================================================
    ## Temp:
    #ret = model_time_measurement(6000, kernel_num = 5, repetitions_num=1, run_sparse = False, run_ss = False, run_gp = True)
    #print(ret)

    
