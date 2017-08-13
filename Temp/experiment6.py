# -*- coding: utf-8 -*-
"""

Experiment 6

@author: agrigori
"""

import os
import numpy as np
import time
from collections import Iterable
import scipy as sp
import scipy.io as io
import warnings


import matplotlib.pyplot as plt
#import sksparse.cholmod as cholmod
#import GPy.inference.latent_function_inference.ss_sparse_inference as ss
#import ss_sparse_inference as ss

from GPy.inference.latent_function_inference import ss_sparse_inference
ss = ss_sparse_inference.sparse_inference

import GPy

def load_data_mine():
    """
    Load data for the case where error was.
    """    
    
    data_file_path= '/home/agrigori/Programming/python/Sparse GP/CO2_data/co2_weekly_init_clean.csv'
    #data_file_path= '/home/agrigori/Programming/python/Sparse GP/CO2_data/co2_weekly_mlo.txt'
    
    #import pdb; pdb.set_trace()
    data = np.loadtxt(data_file_path); 
    data = data[ np.where(data[:,1] > 0)[0] ,:] # get rid of missing values
    
    y_data = data[:,1]; y_data.shape = (y_data.shape[0],1)
    x_data = data[:,0]; x_data.shape = (x_data.shape[0],1)
    
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    #x_data = x_data - 1974
    
    return x_data, y_data

def load_detrended_data():
    """
    """
    data_file_path= '/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/detrended_alex_data.mat'
    
    res = io.loadmat( data_file_path)
    
    return res['detrended_y']


def load_co2_data_arno():
    """
    Loads data as in Arnos code
    """
    
    files_location_prefix = '/home/agrigori/Programming/python/Sparse GP/CO2_data'

        
    import pdb; pdb.set_trace()
    
    month_data = np.loadtxt(os.path.join(files_location_prefix, 'co2_mm_mlo.txt') )
    week_data = np.loadtxt( os.path.join(files_location_prefix, 'co2_weekly_mlo.txt') )
    
    # Get red of missing values ->
    month_data = month_data[ np.where(month_data[:,3] > 0)[0] ,:] # get rid of missing values. 3-rd column is CO2 conc.
    week_data = week_data[ np.where(week_data[:,4] > 0)[0] ,:] # get rid of missing values. 4-th column is CO2 conc.
    # Get red of missing values <-
    
    # Combine monthly and weekly data ->
    t0 = np.min( week_data[:, 3] ) # find min time in weekly data.
    inds = np.where(month_data[:,2] < t0)[0] # find earlier data of monthly data
    
    x_data = np.concatenate( (month_data[inds,2],  week_data[:, 3] )  ); x_data.shape = (x_data.shape[0],1)
    y_data = np.concatenate( (month_data[inds,3],  week_data[:, 4] )  ); y_data.shape = (y_data.shape[0],1)
    # Combine monthly and weekly data <-
    
    # Normalize y ->
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    # Normalize y <-


    # Divite on train and validation ->
    inds_train = x_data[ np.where(x_data[:,0] <2010 )[0] ,:]    
    inds_valid = x_data[ np.where(x_data[:,0] >= 2010 )[0] ,:]
    
    x_train = x_data[inds_train,:]; y_train = y_data[inds_train,:];
    x_valid = x_data[inds_valid,:]; y_valid = y_data[inds_valid,:];
    # Divite on train and validation <-

    return x_train, y_train

def save_ss_model(*ss_matrices):
    """

    """
    prefix_file_save = '/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/'    
    file_name = 'tmp1.mat'
    
    res = {}

    res['pF'] = ss_matrices[0]
    res['pL'] = ss_matrices[1]  
    res['pQc'] = ss_matrices[2]
    res['pH'] = ss_matrices[3]
    res['pPinf'] = ss_matrices[4]    
    res['pP0'] = ss_matrices[5] 
    res['pdF'] = ss_matrices[6]
    res['pdQc'] = ss_matrices[7]
    res['pdPinf'] = ss_matrices[8]
    res['pdP0'] = ss_matrices[9]    
    
    io.savemat(os.path.join(prefix_file_save, file_name), res)
    
def exp_quad_cov_ss(balance=False):
    """
    Export state-space covariance function 
    
    There is a difference between Python and Matlab implementation of RBF kernel
    the difference is (accurding to tests) appear because of different lyapunov equation solver.
    
    Balancing is also different in matlab and in Python. Just balancing function
    works differently, hence the results are not comparable.
    """
    model_params = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')

    var_1_trend = model_params['rbf_magnSigma2']
    ls_1_trend = model_params['rbf_lengthscale']
    
#    var_1_trend = 1e4
#    ls_1_trend = 100
    
    kernel1 = GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend, balance = False, approx_order = 6)
    (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = kernel1.sde()
    
    #import pdb; pdb.set_trace()
    if balance:
        import GPy.models.state_space_main as ssm
        (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = ssm.balance_ss_model(F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0 )
        
    #import pdb; pdb.set_trace()
    return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)

def matern32_cov_ss(balance=False):
    """
    Export state-space covariance function 
    
    """    
    var_1_per = 1.37
    ls_1_per = 140
    
    kernel1 = GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
    (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = kernel1.sde()
    
    if balance:
        import GPy.models.state_space_main as ssm
        (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = ssm.balance_ss_model(F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0 )
    
    #import pdb; pdb.set_trace()
    return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)

def periodic_cov_ss(balance=False):
    """
    Export state-space covariance function 
    
    """    
    var_1_per = 5
    ls_1_per = 2.3 / 2.0 # division by 2 is done because meanings in parameters are different
    period = 1.46    
    
    kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order = 6)
    (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = kernel1.sde()
    
    
    if balance:
        import GPy.models.state_space_main as ssm
        (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = ssm.balance_ss_model(F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0 )
    
    #import pdb; pdb.set_trace()
    return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)

def periodic_cov_ss_opt_params(balance=False):
    """
    Export state-space covariance function 
    
    """
    
    model_params = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
   
#    model.param_array[:] = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale'],
#                                    1.0, model_params['quasi_per_mat32_lengthscale'],  model_params['mat32_inacc_magnSigma2'],
#                                     model_params['mat32_inacc_lengthScale'], model_params['opt_noise'] ] )
   
    var_1_per = float(model_params['per_magnSigma2'])
    ls_1_per = float(model_params['per_lengthscale']) / 2.0 # division by 2 is done because meanings in parameters are different
    period = float(model_params['per_period'][0])     
    
    kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order = 6)
    #import pdb; pdb.set_trace()
    (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = kernel1.sde()
    
    if balance:
        import GPy.models.state_space_main as ssm
        (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = ssm.balance_ss_model(F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0 )
    
    #import pdb; pdb.set_trace()
    return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)
    
def cov_except_RBF_ss(balance=False):
    """
    State space model of covariance     
    """
    
    var_1_per = 5
    ls_1_per = 1.3 / 2 # division by 2 is done because meanings in parameters are different
    period = 1.2    
    kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order = 6)
    
    var_1_per = 1 # does not change
    ls_1_per = 140
    kernel2 = GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
    
    
#    model.ss{1}.lengthScale  = 1;
#    model.ss{1}.magnSigma2   = 5;
#    model.ss{1}.period       = 1; % one year
#    model.ss{1}.N            = 6; # appr. order for periodic part
#    model.ss{1}.nu           = 3/2;
#    model.ss{1}.mN           = 6; # appr. order for squared exponential
#    model.ss{1}.mlengthScale = 140;
#    model.ss{1}.opt          = {'magnSigma2','lengthScale','mlengthScale'};
    
    
    # Short term fluctuations kernel    
    var_1_per = 0.5
    ls_1_per = 1.7
    kernel3 =  GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)    
    
    kernel = kernel1 * kernel2 + kernel3
    
    print( kernel.parameter_names() )
    #import pdb; pdb.set_trace()
    
    (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = kernel.sde()
    
    if balance:
        import GPy.models.state_space_main as ssm
        (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = ssm.balance_ss_model(F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0 )
    
    #import pdb; pdb.set_trace()
    return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)

def cov_except_RBF_ss_opt_params(balance=False):
    """
    Almost the same as before except parameters are optimal
    """

    model_params = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
   
#    model.param_array[:] = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale'],
#                                    1.0, model_params['quasi_per_mat32_lengthscale'],  model_params['mat32_inacc_magnSigma2'],
#                                     model_params['mat32_inacc_lengthScale'], model_params['opt_noise'] ] )

    var_1_per = model_params['per_magnSigma2']
    ls_1_per = model_params['per_lengthscale'] / 2 # division by 2 is done because meanings in parameters are different
    period = model_params['per_period']    
    kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order = 6)
    
    var_1_per = 1 # does not change
    ls_1_per = model_params['quasi_per_mat32_lengthscale']
    kernel2 = GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
    

    # Short term fluctuations kernel    
    var_1_per = model_params['mat32_inacc_magnSigma2']
    ls_1_per = model_params['mat32_inacc_lengthScale']
    kernel3 =  GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)    
    
    kernel = kernel1*kernel2 + kernel3
    
    print( kernel.parameter_names() )
    #import pdb; pdb.set_trace()
    
    (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = kernel.sde()
    
    if balance:
        import GPy.models.state_space_main as ssm
        (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0) = ssm.balance_ss_model(F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0 )
    
    #import pdb; pdb.set_trace()
    return (F, L, Qc, H, Pinf, P0, dF, dQc, dPinf, dP0)
    
def model_data(without_rbf=True):
    """
    Load the data from the matlab and fit the propper covariance
    Either using or not using RBF covariance. Matlab data is exported either using or not using
    rbf as well.
    
    """
    #!!! Download data from matlab
    res = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/tmp2.mat')

    ml_x_train = res['to']
    ml_x_new = res['x_pred'] #- 1974
    matlab_y_train = res['y_train']
    
    # !!! initial values are any because we are loading right model parameters
    var_1_per = 1
    ls_1_per = 1 / 2 # division by 2 is done because meanings in parameters are different
    period = 1    
    kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order = 6)
    
    var_1_per = 1 # does not change
    ls_1_per = 1
    kernel2 = GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
    
    # Short term fluctuations kernel
    var_1_per = 1
    ls_1_per = 1
    kernel3 =  GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
    
    # RBF kernel. Not used if the parameter without_rbf = True 
    var_1_per = 1
    ls_1_per = 1
    kernel4 =  GPy.kern.sde_RBF(1, variance=var_1_per, lengthscale=ls_1_per, balance=False, approx_order = 6)    
    
    # RBF kernel. Not used if the parameter 
    
    if without_rbf:
        kernel = kernel1 * kernel2 + kernel3
    else:
        kernel = kernel1 * kernel2 + kernel3 + kernel4
        
    x_train, y_data = load_data_mine()
    x_train = x_train #- 1974
    x_train = ml_x_train # from matlab
    y_train = load_detrended_data()
    y_train  = matlab_y_train # from matlab

    model_params = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
    model = GPy.models.StateSpace( x_train, y_train, kernel=kernel, noise_var=1.0, balance=False, kalman_filter_type = 'svd')
    
    #import pdb; pdb.set_trace()
    
    # param names:
    #['sum.mul.std_periodic.variance', 'sum.mul.std_periodic.period', 'sum.mul.std_periodic.lengthscale', 
    #'sum.mul.Mat32.variance', 'sum.mul.Mat32.lengthscale', 'sum.Mat32.variance', 'sum.Mat32.lengthscale', 
    #'Gaussian_noise.variance']
    
    if without_rbf:
        # param names:
        #['sum.mul.std_periodic.variance', 'sum.mul.std_periodic.period', 'sum.mul.std_periodic.lengthscale', 
        #'sum.mul.Mat32.variance', 'sum.mul.Mat32.lengthscale', 'sum.Mat32.variance', 'sum.Mat32.lengthscale', 
        #'Gaussian_noise.variance']
        model.param_array[:] = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale']/2,
                                    1.0, model_params['quasi_per_mat32_lengthscale'],  model_params['mat32_inacc_magnSigma2'],
                                     model_params['mat32_inacc_lengthScale'], model_params['opt_noise'] ] )
    else:
        # param names: model.parameter_names()
        #['sum.mul.std_periodic.variance', 'sum.mul.std_periodic.period', 
        #'sum.mul.std_periodic.lengthscale', 'sum.mul.Mat32.variance', 
        #'sum.mul.Mat32.lengthscale', 'sum.Mat32.variance', 'sum.Mat32.lengthscale', 
        #'sum.rbf.variance', 'sum.rbf.lengthscale', 'Gaussian_noise.variance']
        
        model.param_array[:] = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale']/2,
                                    1.0, model_params['quasi_per_mat32_lengthscale'],  model_params['mat32_inacc_magnSigma2'],
                                     model_params['mat32_inacc_lengthScale'], 
            model_params['rbf_magnSigma2'], model_params['rbf_lengthscale'], model_params['opt_noise'] ] )
        
    rr = model.kern.sde()
    save_ss_model(*rr)     
                            
    years_to_predict = 8
    step = np.mean( np.diff(x_train[:,0]) )
    
    x_new = x_train[-1,0] + np.arange( step, years_to_predict,  step ); x_new.shape = (x_new.shape[0],1)
    x_new = np.vstack( (x_train, x_new)) # combine train and test data
    x_new = ml_x_new
    
    ssm_mean, ssm_var = model.predict(x_new, include_likelihood=False)    

    plt.figure(1)
    #plt.title('Electricity Consumption Data', fontsize=30)    
    plt.plot( x_train, y_train, 'g-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean, 'b-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean+np.sqrt(ssm_var), 'r--', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean-np.sqrt(ssm_var), 'r--', label='Data',linewidth=1, markersize=5)
    #plt.xlabel('Time (Hours)', fontsize=25)
    #plt.ylabel('Normalized Value', fontsize=25)
    #plt.legend(loc=2)
    plt.show()    
    
    
    print(model)
    #import pdb; pdb.set_trace()
    return x_new, y_train, ssm_mean, ssm_var


def compare_matlab_and_python_predictions(x_new, y_new, ssm_mean, ssm_var):
    """
    
    """

    res = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/tmp2.mat')

    ml_x_new = res['x_pred'] #- 1974
    ml_y_new = res['y_train']
    ml_ssm_mean = res['meanf'].T
    ml_ssm_var = res['Varf'].T # 
    
    #import pdb; pdb.set_trace()
    print('Max X diff %f', np.max( np.abs( ml_x_new - x_new) ))    
    print('Max Y diff %f', np.max( np.abs( ml_y_new - y_new) ))
    
    print('Max MEAN diff %f', np.max( np.abs( ml_ssm_mean - ssm_mean) ))    
    print('Max VAR diff %f', np.max( np.abs( ml_ssm_var - ssm_var) ))
    
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(x_new, ssm_mean, 'r.-', ml_x_new, ml_ssm_mean,'b.-')
    plt.subplot(1,2,2)
    plt.plot(x_new, ssm_var, '.-r', ml_x_new, ml_ssm_var,'-b')
    plt.show()

    plt.figure(2)
    plt.plot( ml_x_new - x_new )
    plt.show()
    
    #import pdb; pdb.set_trace()

def experiment6_action4(hyperparameters='my_ex6_optimal'):
    """
    State-space predictions my orig data.
    
    Copies the first submision experiment6 action 4. It was working improperly
    earlier. What is now? - Works
    
    """    
    
    # Data ->
    data_file_path= '/home/agrigori/Programming/python/Sparse GP/CO2_data/co2_weekly_init_clean.csv'
    results_filex_prefix = '/home/agrigori/Programming/python/Sparse GP/Experiemnts/Results'
    
    data = np.loadtxt(data_file_path); 
    data = data[ np.where(data[:,1] > 0)[0] ,:] # get rid of missing values
    
    y_data = data[:,1]; y_data.shape = (y_data.shape[0],1)
    x_data = data[:,0]; x_data.shape = (x_data.shape[0],1)
    
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_data = x_data - 1974
    # Data <-
    
    if hyperparameters == 'my_ex6_optimal':
        var_1_trend = 1.0
        ls_1_trend = 200
        
        per_per = 1
        per_var = 1.0 # fixed
        per_ls = 1
        var_2_quasi = 1.0
        ls_2_quasi = 1.0
        
        var_3_quasi = 1.0
        ls_3_quasi = 100.0
        
        noise_var = 0.1
    
        kernel = GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend) +\
                  GPy.kern.sde_StdPeriodic(1, variance=per_var, period = per_per, lengthscale=per_ls)*\
                  GPy. kern.sde_Matern32(1, variance=var_2_quasi, lengthscale=ls_2_quasi) +\
                  GPy. kern.sde_Matern32(1, variance=var_3_quasi, lengthscale=ls_3_quasi)
    
        kernel.mul.std_periodic.variance.fix()
        kernel.mul.std_periodic.period.fix()
        
        result_dict = io.loadmat(os.path.join(results_filex_prefix,'ex6_params'))
        params = result_dict['params'].squeeze()
        
        kernel[:] = params[0:-1]
        noise_var = params[-1]

        model = GPy.models.StateSpace( x_data, y_data, kernel=kernel, noise_var=noise_var, balance=False, kalman_filter_type = 'regular')        
    elif hyperparameters == 'arno_optimal':
        model_params = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
        
        var_1_per = 1
        ls_1_per = 1 / 2 # division by 2 is done because meanings in parameters are different
        period = 1    
        kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order = 6)
        
        var_1_quasi = 1 # does not change
        ls_1_quasi = 1
        kernel2 = GPy.kern.sde_Matern32(1, variance=var_1_quasi, lengthscale=ls_1_quasi)
        
        # Short term fluctuations kernel
        var_1_short = 1
        ls_1_short = 1
        kernel3 =  GPy.kern.sde_Matern32(1, variance=var_1_short, lengthscale=ls_1_short)
        
        # RBF kernel. Not used if the parameter without_rbf = True 
        var_1_trend = 1
        ls_1_trend = 1
        kernel4 =  GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend, balance=False, approx_order = 6)    
        
        # RBF kernel. Not used if the parameter 
        
        
        kernel = kernel1 * kernel2 + kernel3 + kernel4
        
        model = GPy.models.StateSpace( x_data, y_data, kernel=kernel, noise_var=1.0, balance=False, kalman_filter_type = 'svd')
    
        model.param_array[:] = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale']/2,
                                    1.0, model_params['quasi_per_mat32_lengthscale'],  model_params['mat32_inacc_magnSigma2'],
                                     model_params['mat32_inacc_lengthScale'], 
            model_params['rbf_magnSigma2'], model_params['rbf_lengthscale'], model_params['opt_noise'] ] )
        print(model.objective_function())
    print(model)
    # predict ->
    years_to_predict = 8
    step = np.mean( np.diff(x_data[:,0]) )
    
    x_new = x_data[-1,0] + np.arange( step, years_to_predict,  step ); x_new.shape = (x_new.shape[0],1)
    x_new = np.vstack( (x_data, x_new)) # combine train and test data
    
    ssm_mean, ssm_var = model.predict(x_new, include_likelihood=False)
    # predict <-
    
        

    plt.figure(1)
    #plt.title('Electricity Consumption Data', fontsize=30)    
    plt.plot( x_data, y_data, 'g-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean, 'b-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean+np.sqrt(ssm_var), 'r--', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean-np.sqrt(ssm_var), 'r--', label='Data',linewidth=1, markersize=5)
    #plt.xlabel('Time (Hours)', fontsize=25)
    #plt.ylabel('Normalized Value', fontsize=25)
    #plt.legend(loc=2)
    plt.show()    

def experiment6_action3(hyperparameters='my_ex6_optimal'):
    """
    Copies the first submision experiment6 action 4. It was working improperly
    earlier. What is now?
    
    'my_ex6_optimal' are probably somehow bad, did not work ok, however did not 
    try much as well. 
    
    'arno_optimal' - works well. Values of max_cond_number start from 1e+15. (1e+16 gives error)
            1e+14 - visually indistinguishable from ss run. If we decrease
            consitional number then variance expands, mean visually is not affected.
            
    """
        
    import GPy.models.ss_sparse_model as ss_sparse_model
    
    # Data ->
    data_file_path= '/home/agrigori/Programming/python/Sparse GP/CO2_data/co2_weekly_init_clean.csv'
    results_filex_prefix = '/home/agrigori/Programming/python/Sparse GP/Experiemnts/Results'
    
    data = np.loadtxt(data_file_path); 
    data = data[ np.where(data[:,1] > 0)[0] ,:] # get rid of missing values
    
    y_data = data[:,1]; y_data.shape = (y_data.shape[0],1)
    x_data = data[:,0]; x_data.shape = (x_data.shape[0],1)
    
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_data = x_data - 1974
    # Data <-
    
    if hyperparameters == 'my_ex6_optimal':
        var_1_trend = 1.0
        ls_1_trend = 200
        
        per_per = 1
        per_var = 1.0 # fixed
        per_ls = 1
        var_2_quasi = 1.0
        ls_2_quasi = 1.0
        
        var_3_quasi = 1.0
        ls_3_quasi = 100.0
        
        noise_var = 0.1
    
        kernel = GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend) +\
                  GPy.kern.sde_StdPeriodic(1, variance=per_var, period = per_per, lengthscale=per_ls)*\
                  GPy. kern.sde_Matern32(1, variance=var_2_quasi, lengthscale=ls_2_quasi) +\
                  GPy. kern.sde_Matern32(1, variance=var_3_quasi, lengthscale=ls_3_quasi)
    
        kernel.mul.std_periodic.variance.fix()
        kernel.mul.std_periodic.period.fix()
        
        result_dict = io.loadmat(os.path.join(results_filex_prefix,'ex6_params'))
        params = result_dict['params'].squeeze()
        
        kernel[:] = params[0:-1]
        noise_var = params[-1]
        model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=noise_var, balance=False, p_Inv_jitter=1e-14 )
        
    elif hyperparameters == 'arno_optimal':
        model_params = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
        
        var_1_per = 1
        ls_1_per = 1 / 2 # division by 2 is done because meanings in parameters are different
        period = 1    
        kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order = 6)
        
        var_1_per = 1 # does not change
        ls_1_per = 1
        kernel2 = GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
        
        # Short term fluctuations kernel
        var_1_per = 1
        ls_1_per = 1
        kernel3 =  GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
        
        # RBF kernel. Not used if the parameter without_rbf = True 
        var_1_per = 1
        ls_1_per = 1
        kernel4 =  GPy.kern.sde_RBF(1, variance=var_1_per, lengthscale=ls_1_per, balance=False, approx_order = 6)    
        
        # RBF kernel. Not used if the parameter 
        
        
        kernel = kernel1 * kernel2 + kernel3 + kernel4
        
        model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=1.0, balance=False, largest_cond_num=1e+14, regularization_type=2)
    
        model.param_array[:] = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale']/2,
                                    1.0, model_params['quasi_per_mat32_lengthscale'],  model_params['mat32_inacc_magnSigma2'],
                                     model_params['mat32_inacc_lengthScale'], 
            model_params['rbf_magnSigma2'], model_params['rbf_lengthscale'], model_params['opt_noise'] ] )

    print(model)
    # predict ->
    years_to_predict = 8
    step = np.mean( np.diff(x_data[:,0]) )
    
    x_new = x_data[-1,0] + np.arange( step, years_to_predict,  step ); x_new.shape = (x_new.shape[0],1)
    x_new = np.vstack( (x_data, x_new)) # combine train and test data
    
    ssm_mean, ssm_var = model.predict(x_new, include_likelihood=False, largest_cond_num=1e+14, regularization_type=2)
    # predict <-
    
    plt.figure(2)
    #plt.title('Electricity Consumption Data', fontsize=30)    
    plt.plot( x_data, y_data, 'g-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean, 'b-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean+np.sqrt(ssm_var), 'r--', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean-np.sqrt(ssm_var), 'r--', label='Data',linewidth=1, markersize=5)
    #plt.xlabel('Time (Hours)', fontsize=25)
    #plt.ylabel('Normalized Value', fontsize=25)
    #plt.legend(loc=2)
    plt.show()    

def experiment6_action1(hyperparameters='arno_start', p_model='ss', optim = 'bfgs', bound_qp_var=False):
    """
    Copies the first submision experiment6 action 1. Training of hyper parameters.
    It was working improperly
    earlier. What is now? - Working.
    """ 
    
    import GPy.models.ss_sparse_model as ss_sparse_model
    
    # Data ->
    data_file_path= '/home/agrigori/Programming/python/Sparse GP/CO2_data/co2_weekly_init_clean.csv'
    results_filex_prefix = '/home/agrigori/Programming/python/Sparse GP/Experiemnts/Results'
    
    data = np.loadtxt(data_file_path); 
    data = data[ np.where(data[:,1] > 0)[0] ,:] # get rid of missing values
    
    y_data = data[:,1]; y_data.shape = (y_data.shape[0],1)
    x_data = data[:,0]; x_data.shape = (x_data.shape[0],1)
    
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_data = x_data - 1974
    # Data <-
    
    if hyperparameters == 'arno_start':
        # Start hyper parameters which were used in the 
        var_1_trend = 1e4
        ls_1_trend = 100.0
        
        per_per = 1 # fixed
        per_var = 5.0
        per_ls = 1
        var_2_quasi = 1.0
        ls_2_quasi = 140
        
        var_3_quasi = 0.5
        ls_3_quasi = 1
        
        noise_var = 1.0
    
        kernel =  GPy.kern.sde_StdPeriodic(1, variance=per_var, period = per_per, lengthscale=per_ls, balance=False, approx_order = 6)*\
                  GPy.kern.sde_Matern32(1, variance=var_2_quasi, lengthscale=ls_2_quasi) +\
                  GPy.kern.sde_Matern32(1, variance=var_3_quasi, lengthscale=ls_3_quasi) +\
                  GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend, balance=False, approx_order = 6)
                  
        kernel.mul.Mat32.variance.fix()
        kernel.mul.std_periodic.period.fix()
        if bound_qp_var:
            kernel.mul.std_periodic.lengthscale.constrain_bounded(0.2, 20000)
#        result_dict = io.loadmat(os.path.join(results_filex_prefix,'ex6_params'))
#        params = result_dict['params'].squeeze()
#        
#        kernel[:] = params[0:-1]
#        noise_var = params[-1]
        if p_model == 'ss':            
            model = GPy.models.StateSpace( x_data, y_data, kernel=kernel, noise_var=noise_var, balance=False, kalman_filter_type = 'svd')
        elif p_model == 'sparse':
            # Regularization assume regularization type 2
            model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=1.0, balance=False, largest_cond_num=1e+11, regularization_type=2)
            # optim scg: Kernel: std_per approx order=6 bal=False, rbf_approx_order=6, bal=False. Model:  balance=False, largest_cond_num=1e+12, regularization_type=2.
            # optim bfgs: Kernel: std_per approx order=6 bal=False, rbf_approx_order=5, bal=False. Model:  balance=False, largest_cond_num=1e+11, regularization_type=2.
        else:
            raise ValueError("Wrong Parameter 1")
    elif hyperparameters == 'arno_optimal':
        model_params = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
        
        var_1_per = 1
        ls_1_per = 1 / 2 # division by 2 is done because meanings in parameters are different
        period = 1    
        kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order = 6)
        
        var_1_per = 1 # does not change
        ls_1_per = 1
        kernel2 = GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
        
        # Short term fluctuations kernel
        var_1_per = 1
        ls_1_per = 1
        kernel3 =  GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
        
        # RBF kernel. Not used if the parameter without_rbf = True 
        var_1_per = 1
        ls_1_per = 1
        kernel4 =  GPy.kern.sde_RBF(1, variance=var_1_per, lengthscale=ls_1_per, balance=False, approx_order = 6)    
        
        # RBF kernel. Not used if the parameter 
        
        
        kernel = kernel1 * kernel2 + kernel3 + kernel4
        
        if p_model == 'ss':
            model = GPy.models.StateSpace( x_data, y_data, kernel=kernel, noise_var=1.0, balance=False, kalman_filter_type = 'regular')
        elif p_model == 'sparse':
            # Regularization assume regularization type 2
            
            model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=1.0, balance=False, largest_cond_num=1e+15, regularization_type=2) 
            # Parameters of runs: 
            # optim bfgs: Kernel: std_per approx order=6 bal=False, rbf_approx_order=6, bal=False. Model: balance=False, largest_cond_num=1e+15, regularization_type=2
            # optim scg: Kernel: std_per approx order=6 bal=False, rbf_approx_order=6, bal=False. Model: balance=False, largest_cond_num=1e+13, regularization_type=2
        else:
            raise ValueError("Wrong Parameter 2")
        
        tt = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale']/2,
                                    1.0, model_params['quasi_per_mat32_lengthscale'],  model_params['mat32_inacc_magnSigma2'],
                                     model_params['mat32_inacc_lengthScale'], 
            model_params['rbf_magnSigma2'], model_params['rbf_lengthscale'], model_params['opt_noise'] ] )
            
        #print(tt)
        model.param_array[:] = tt
        
        #import pdb; pdb.set_trace()
        model.kern.mul.Mat32.variance.fix()
        model.kern.mul.std_periodic.period.fix()
        if bound_qp_var:
            model.kern.mul.std_periodic.lengthscale.constrain_bounded(0.2, 20000)
        #model.kern.mul.std_periodic.
        print(model)
        
    class observer(object):
        def __init__(self):
            """
            
            """
            
            self.opt_obj_grads = None
            self.opt_gradients = None
            self.opt_params = None
            self.opt_obj_funcs = None
            
        def notif_callable(self, me, which=None):
            """
            Description
            """
    
            #import pdb; pdb.set_trace()

            # me.obj_grads            
            if isinstance(self.opt_obj_grads , np.ndarray): # previous array          
                self.opt_obj_grads = np.vstack( (self.opt_obj_grads, me.obj_grads) )                
            else:
                rr = me.obj_grads
                if isinstance(self.opt_obj_grads , list):
                    if isinstance(rr, np.ndarray):
                        tt = np.empty( (len(self.opt_obj_grads), rr.shape[0]) ) * np.nan
                        self.opt_obj_grads = np.vstack( (tt, rr) )
                    else:     
                        self.opt_obj_grads.append( (rr if rr is not None else np.nan) )
                else:
                    if isinstance(rr , np.ndarray):
                        self.opt_obj_grads = rr
                    else:
                        self.opt_obj_grads = [  (rr if (rr is not None) else np.nan), ]
    
            
            # me.gradient            
            if isinstance(self.opt_gradients , np.ndarray): # previous array          
                self.opt_gradients = np.vstack( (self.opt_gradients, me.gradient) )
            else:
                rr = me.gradient # same as me.objective_function_gradients()              
                if isinstance(self.opt_gradients , list):
                    if isinstance(rr, np.ndarray):
                        tt = np.empty( (len(self.opt_gradients), rr.shape[0]) ) * np.nan
                        self.opt_gradients = np.vstack( (tt, rr) )
                    else:     
                        self.opt_gradients.append( (rr if rr is not None else np.nan) )
                else:
                    if isinstance(rr , np.ndarray):
                        self.opt_gradients = rr
                    else:
                        self.opt_gradients = [  (rr if (rr is not None) else np.nan), ]
            
            # me.param_array            
            if isinstance(self.opt_params , np.ndarray): # previous array          
                self.opt_params = np.vstack( (self.opt_params, me.param_array) )
            else:
                rr = me.param_array # same as me.objective_function_gradients()              
                if isinstance(self.opt_params , list):
                    if isinstance(rr, np.ndarray):
                        tt = np.empty( (len(self.opt_params), rr.shape[0]) ) * np.nan
                        self.opt_params = np.vstack( (tt, rr) )
                    else:     
                        self.opt_params.append( (rr if rr is not None else np.nan) )
                else:
                    if isinstance(rr , np.ndarray):
                        self.opt_params = rr
                    else:
                        self.opt_params = [  (rr if (rr is not None) else np.nan), ]
        
            if self.opt_obj_funcs is None: # first iteration
                self.opt_obj_funcs = [me.objective_function(),] if (me.objective_function() is not None) else [np.nan,]
            else:
                self.opt_obj_funcs.append( me.objective_function() )
        
        def save(self, optim_save_path, optim_file_name, save= False):
            """
            Saves optim data to file
            """
            
            result_dict = {}
            result_dict['opt_obj_grads'] = self.opt_obj_grads
            result_dict['opt_gradients'] = self.opt_gradients
            result_dict['opt_params'] = self.opt_params
            result_dict['opt_obj_funcs'] = np.array(self.opt_obj_funcs)
            
            optim_file_name = optim_file_name + '__optim_history'
            if save:
                io.savemat(os.path.join(optim_save_path, optim_file_name), result_dict)
            
            return result_dict
            
            
    oo = observer()
    model.add_observer( oo, oo.notif_callable )
    
    if optim == 'bfgs':
        # L-BFGS-B optimization ->
        #' factr' - criteria for stoping. If eps * factr < (f_k - f_(k+1)) then stop
        # default: 1e7    
        #
        # pgtol : float, optional
        #    The iteration will stop when
        #    ``max{|proj g_i | i = 1, ..., n} <= pgtol``
        #    where ``pg_i`` is the i-th component of the projected gradient.
        # default: 1e-5
        model.optimize(optimizer ='lbfgsb', bfgs_factor = 1e12, gtol=1e-3, messages=True)
        # L-BFGS-B optimization <-
    elif  optim == 'scg':
        # SCG optimization ->
        # xtol: controls reduction in step size
        #     default: 1e-6
        # ftol: controls reduction in objective                   
        #     default: 1e-6
        # gtol: controls reduction in gradients
        #     default: 1e-5
        
        
        model.optimize(optimizer ='scg', xtol = 1e-3, ftol=1e-3,  gtol = 1e-3, messages=True)
        # SCG optimization <-
    else:
        raise ValueError("Wrong Optimizer name")
        #model.optimize(optimizer ='scg', messages=True)
    print(model)
    
    result_dict = {}
    result_dict['params'] = model.param_array[:]
    result_dict.update( oo.save('', '', save= False) )
    save_file_name = '1_ex6_new_' + p_model+ '_' + hyperparameters + '_' + optim + '_' + '_qp_bound_' +str(bound_qp_var)
    print(results_filex_prefix)
    print(save_file_name)    
    io.savemat(os.path.join(results_filex_prefix, save_file_name), result_dict)
    
    return model, oo


def evaluate_optimization(p_model='ss'):
    """
    1) Order of RBF kernel matters a lot. If it is larger the model can be unstable.
       Balancing of RBF model may also matter.
    
    2) Also the order of kernels matter. Changing from RBF the first or the last
    matter. - this is strange and undesirable of course. Actually, this does not make
    much sence since parameters from optimization are assigned in a certain order and
    changing this order spoil all parameter values. This was a mistake.
    
    3) Sparse model:
        '1_ex6_new_ss_arno_optimal_bfgs__qp_bound_True': Work OK. min_reg=1e+11
            No other modifications helped.
        
        '1_ex6_new_ss_arno_optimal_scg__qp_bound_True': Work OK. Variances are very small.
            min_reg=1e+15. Works also with different orders of RBF and periodic.
            
        '1_ex6_new_ss_arno_start_bfgs__qp_bound_True': Not very OK. The mean is 
            also distorted by regularization. This is the first time this effect is observed!
            min_reg=1e+11
            If we change the order of periodic kernel 6->5 then the mean distorsion effect
            gets much smaller. Other changes did not give any better results. Min reg is 
            also not increased.
            
        '1_ex6_new_ss_arno_start_scg__qp_bound_True': Work OK. min_reg=1e+15
            However lower min_reg do not work. E.g. min_reg=1e+13 produce an error.
            min_reg=1e+11 again work and result is ok, min_reg=1e+10 and 
            min_reg=1e+9 again does not work. 
            min_reg=1e+8 - works but mean and variance are distorted, probably
            because regularization type 1 turns on.
            
            Interesting case! Unclear how regularization work then!
            Errors might appear in both Cholmod and Numpy.
            E.g. (min_reg=1e+13, RBF order=6, Periodic order=6, all balances=False) gives Cholmod error.
            (min_reg=1e+13, RBF order=5, Periodic order=5, all balances=False) 
            Gives Numpy error in computing Lambdas.
            
            
    """
    file_name = '1_ex6_new_ss_arno_start_scg__qp_bound_True'
    file_path = '/home/agrigori/Programming/python/Sparse GP/Experiemnts/Results'

    results_dict = io.loadmat( os.path.join(file_path, file_name) )

     # Data ->
    data_file_path= '/home/agrigori/Programming/python/Sparse GP/CO2_data/co2_weekly_init_clean.csv'
    results_filex_prefix = '/home/agrigori/Programming/python/Sparse GP/Experiemnts/Results'
    
    data = np.loadtxt(data_file_path); 
    data = data[ np.where(data[:,1] > 0)[0] ,:] # get rid of missing values
    
    y_data = data[:,1]; y_data.shape = (y_data.shape[0],1)
    x_data = data[:,0]; x_data.shape = (x_data.shape[0],1)
    
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_data = x_data - 1974
    # Data <-
    
    # predict data->
    years_to_predict = 8
    step = np.mean( np.diff(x_data[:,0]) )
    
    x_new = x_data[-1,0] + np.arange( step, years_to_predict,  step ); x_new.shape = (x_new.shape[0],1)
    x_new = np.vstack( (x_data, x_new)) # combine train and test data
    
    #ssm_mean, ssm_var = model.predict(x_new, include_likelihood=False, largest_cond_num=1e+15, regularization_type=2)
    # predict data<-
    
    # Model ->
    var_1_per = 1
    ls_1_per = 1 / 2 # division by 2 is done because meanings in parameters are different
    period = 1    
    kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order=5)
    
    var_1_quasi = 1 # does not change
    ls_1_quasi = 1
    kernel2 = GPy.kern.sde_Matern32(1, variance=var_1_quasi, lengthscale=ls_1_quasi)
    
    # Short term fluctuations kernel
    var_1_short = 1
    ls_1_short = 1
    kernel3 =  GPy.kern.sde_Matern32(1, variance=var_1_short, lengthscale=ls_1_short)
    
    # RBF kernel. Not used if the parameter without_rbf = True 
    var_1_trend = 1
    ls_1_trend = 1
    kernel4 =  GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend, balance=False, approx_order=5)    
    
    # RBF kernel. Not used if the parameter 
    
    kernel = kernel1 * kernel2 + kernel3 + kernel4 # keep this order because it has been used during optimization 
    #kernel = kernel4 + kernel1 * kernel2 + kernel3
    
#    kernel = GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend, balance=False, approx_order = 6) +\
#              GPy.kern.sde_StdPeriodic(1, variance=var_1_per, period = period, lengthscale=ls_1_per)*\
#              GPy.kern.sde_Matern32(1, variance=var_1_quasi, lengthscale=ls_1_quasi) +\
#              GPy.kern.sde_Matern32(1, variance=var_1_short, lengthscale=ls_1_short)
    
   
#    kernel =  GPy.kern.sde_StdPeriodic(1, variance=var_1_per, period = period, lengthscale=ls_1_per)*\
#              GPy.kern.sde_Matern32(1, variance=var_1_quasi, lengthscale=ls_1_quasi) +\
#              GPy.kern.sde_Matern32(1, variance=var_1_short, lengthscale=ls_1_short) +\
#              GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend, balance=False, approx_order = 6)
              
    if p_model == 'ss':            
        model = GPy.models.StateSpace( x_data, y_data, kernel=kernel, noise_var=1.0, balance=False, kalman_filter_type = 'svd')
    elif p_model == 'sparse':
        # Regularization assume regularization type 2
        import GPy.models.ss_sparse_model as ss_sparse_model
        model1 = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=1.0, balance=False, largest_cond_num=1e+14, regularization_type=2) # Arno optimal parameters
        model2 = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel.copy(), noise_var=1.0, balance=False, largest_cond_num=1e+10, regularization_type=2) # Parameters from optimization 
    else:
        raise ValueError("Wrong Parameter 1")
    # Model <-
        
    # Arno optimal parameters ->
    model_params = io.loadmat('/home/agrigori/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
    tt = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale']/2,
                                    1.0, model_params['quasi_per_mat32_lengthscale'],  model_params['mat32_inacc_magnSigma2'],
                                     model_params['mat32_inacc_lengthScale'], 
            model_params['rbf_magnSigma2'], model_params['rbf_lengthscale'], model_params['opt_noise'] ] )
    #import pdb; pdb.set_trace()
    #print(tt)
    model1.param_array[:] = tt
    print(model1)
    #print(model1.objective_function())    
    mean1, var1 = model1.predict(x_new, include_likelihood=False) #, largest_cond_num=1e+15, regularization_type=2)    
    #_helper_ss_sparse_compare_gradients(x_data, y_data, kernel, tt, p_step=1e-4)    
    # Arno optimal parameters <-
    
    # Optmim optimal (parameters from optimization) ->
    params = results_dict['params'] 
    #model = GPy.models.StateSpace( x_data, y_data, kernel=kernel, noise_var=1.0, balance=False, kalman_filter_type = 'svd')
    model2.randomize()
    model2.param_array[:] = params
    print(model2)
    mean2, var2 = model2.predict(x_new, include_likelihood=False) #, largest_cond_num=1e+15, regularization_type=2)
    
    #_helper_ss_sparse_compare_gradients(x_data, y_data, kernel, params, p_step=1e-4)
    
    # Optmim optimal (parameters from optimization) <-

    # Plot ->
    plt.figure(2)
    #plt.title('Electricity Consumption Data', fontsize=30)
    plt.subplot(1,2,1)    
    plt.plot( x_data, y_data, 'g-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, mean1, 'b-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, mean1+np.sqrt(var1), 'r--', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, mean1-np.sqrt(var1), 'r--', label='Data',linewidth=1, markersize=5)
    
    plt.subplot(1,2,2)    
    plt.plot( x_data, y_data, 'g-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, mean2, 'b-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, mean2+np.sqrt(var2), 'r--', label='Data',linewidth=1, markersize=5)
    plt.plot( x_new, mean2-np.sqrt(var2), 'r--', label='Data',linewidth=1, markersize=5)
    #plt.xlabel('Time (Hours)', fontsize=25)
    #plt.ylabel('Normalized Value', fontsize=25)
    #plt.legend(loc=2)
    
    plt.show()

    # Plot <-
def _helper_ss_sparse_compare_gradients(x_data, y_data, kernel, p_param_vector, p_step=1e-6):
    """
    
    """
    
    k1 = kernel.copy()
    ss_model = GPy.models.StateSpace( x_data, y_data, kernel=k1, noise_var=1.0, balance=False, kalman_filter_type = 'svd')
    ss_model.param_array[:] = p_param_vector
    ss_model.parameters_changed()
    print(ss_model)
    print(ss_model.checkgrad(verbose=True,step=p_step))
    
    k2 = kernel.copy()
    import GPy.models.ss_sparse_model as ss_sparse_model
    sparse_model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,k2, noise_var=1.0, balance=False, largest_cond_num=1e+12, regularization_type=2) 
    sparse_model.param_array[:] = p_param_vector
    sparse_model.parameters_changed()
    print(sparse_model)
    print(sparse_model.checkgrad(verbose=True,step=p_step))
    
     
if __name__ == "__main__":
    #load_co2_data_arno()
    #rr = exp_quad_cov_ss(balance=False)
    #rr = matern32_cov_ss(balance=True)
    
    #rr = periodic_cov_ss(balance=True)
    #rr = cov_except_RBF_ss(balance=True)
    #rr = periodic_cov_ss_opt_params(balance=True)
    #rr = cov_except_RBF_ss_opt_params(balance=True)
    #save_ss_model(*rr)


    #x_new, y_new, ssm_mean, ssm_var = model_data(without_rbf=False)
    #compare_matlab_and_python_predictions(x_new, y_new, ssm_mean, ssm_var)
    #experiment6_action4(hyperparameters='my_ex6_optimal')
    #experiment6_action4(hyperparameters='arno_optimal') # state-space model
    #experiment6_action3(hyperparameters='arno_optimal') # sparse model
    #warnings.simplefilter("always")
    #(mm, oo) = experiment6_action1(hyperparameters='arno_start', p_model='ss', optim = 'scg', bound_qp_var=True)
    
    #evaluate_optimization(p_model='sparse')
    
#    with warnings.catch_warnings(record=False):
#        warnings.simplefilter("always")
#        experiment6_action1(hyperparameters='arno_optimal')
#        import pdb; pdb.set_trace()
#        print(w)