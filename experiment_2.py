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

from GPy.inference.latent_function_inference import ss_sparse_inference

import GPy

# Global variables ->
#data_file_path= '/u/85/agrigori/unix/Programming/python/Sparse GP/CO2_data/co2_weekly_init_clean.csv'
#results_files_prefix = '/u/85/agrigori/unix/Programming/python/Sparse GP/Experiemnts/Results'

data_file_path = '/home/alex/Programming/python/Sparse GP/CO2_data/co2_weekly_init_clean.csv'
results_files_prefix = '/home/alex/Programming/python/Sparse GP/Experiemnts/Results'
# Global variables <-


def load_data_low_level(data_file_name):
    """
    This is function just loads the data and do basic preprocesing.
    It is called from other higher level functions.
    """    
    
    # Process data ->
    data = np.loadtxt(data_file_name); 
    data = data[ np.where(data[:,1] > 0)[0] ,:] # get rid of missing values
    
    y_data = data[:,1]; y_data.shape = (y_data.shape[0],1)
    x_data = data[:,0]; x_data.shape = (x_data.shape[0],1)
    # Process data <-
    
    return x_data, y_data
    
def load_data(detrend = False, detrend_model = 'gp', detrend_file='ex6_trend_fit_sparse_2.mat',plot=False):
    """
    The function loads the raw data, normalize it (zero mean, unit variance) 
    and optionally detrends it.
    
    Input:
    -------------------------
    detrend: bool
        Whether to detrend or not
    
    detrend_model: which model is used for detrending. Sparse model and StateSpace model
        may be not good for detrending because data dt is too small wrt the trend lengthscale.
    
    detrend_model: string
        Which model is used for detrending.
    
    detrend_file: string
        file_name of the trend parameters. Parameters are found in the function experiment2_fit_trend.
        See also the variable 
    
    plot: bool
        Whether to plot the trend.
    """
    
    #import pdb; pdb.set_trace()
    x_data, y_data = load_data_low_level(data_file_path)
    
    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
    x_data = x_data - 1974 # Start time from 0.
    
    #import pdb; pdb.set_trace()
    if detrend:
        np.random.seed(234) # seed the random number generator
        
        results_dict = io.loadmat( os.path.join(results_files_prefix, detrend_file) )
        
        params = results_dict['params'][0]
        variance_init = float(params[0])
        lengthcale_init = float(params[1])
        noise_var_init = float(params[2])
        
        if detrend_model == 'gp':
            kernel = GPy.kern.RBF(1,variance=variance_init, lengthscale = lengthcale_init)
            kernel.variance.fix()
            kernel.lengthscale.fix()
        
            model = GPy.models.GPRegression(x_data, y_data, kernel, noise_var=noise_var_init)
            model.optimize()
        elif detrend_model == 'ss':
       
            kernel = GPy.kern.sde_RBF(1,variance=variance_init, lengthscale = lengthcale_init,
                                      balance= False, approx_order = 6 )
            kernel.variance.fix()
            kernel.lengthscale.fix()
            
            model = GPy.models.StateSpace(x_data, y_data, kernel, noise_var=noise_var_init, balance=False, kalman_filter_type = 'svd')
            
        elif detrend_model == 'sparse':
            import GPy.models.ss_sparse_model as ss_sparse_model
            kernel = GPy.kern.sde_RBF(1,variance=variance_init, lengthscale = lengthcale_init,
                                      balance= False, approx_order = 6 )
            kernel.variance.fix()
            kernel.lengthscale.fix()
            
            model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=noise_var_init, 
                                                      balance=False, 
                                                      largest_cond_num=1e+20, regularization_type=2)
    #import pdb; pdb.set_trace()
        print('Detrend:')                                              
        print(model)
        (y_pred,var_pred) = model.predict(x_data)
        
        if plot:
            plt.figure(1)
             
            plt.plot( x_data, y_data, 'b.-', label='Data',linewidth=1, markersize=5)
            plt.plot( x_data, y_pred, 'r-', label='Data',linewidth=1, markersize=5)
            plt.plot( x_data, y_pred+np.sqrt(var_pred), 'r--', label='Data',linewidth=1, markersize=5)
            plt.plot( x_data, y_pred-np.sqrt(var_pred), 'r--', label='Data',linewidth=1, markersize=5)
            
            plt.show()   
        
        y_data = y_data - y_pred
        
    return x_data, y_data

def denormalize_data(y_pred_mean, y_pred_std):
    """
    This function denormalizes the data. The function 'load data' normalizes
    the data, so when some model fit is obtained, we need to detrend predictions
    (and training data) to come bace to original scale.
    
    Input:
    ---------------------------
    y_pred_mean: array
        Predicted mean values. 
    
    y_pred_std: array
        Predicted var values.
    """
    
    x_data, y_data = load_data_low_level(data_file_path) 
    
    mean = np.mean(y_data)
    std = np.std(y_data)    
    
#    y_data = (y_data - np.mean(y_data)) / np.std(y_data)
#    x_data = x_data - 1974 # Start time from 0.
    
    y_pred_mean_new = np.squeeze(y_pred_mean)*std + mean 
    y_pred_std_new = np.squeeze(y_pred_std)*std
    
    return y_pred_mean_new, y_pred_std_new, x_data, y_data

def plot_raw_data():
    """
    This function just plots the raw CO_2 data
    """
    
    x_data, y_data = load_data_low_level(data_file_path)
    print(x_data.shape)
    
    # Plot ->
    
    plt.figure(3)
    #(y_min,y_max) = (370,450)    
    #(x_min,x_max) = (2010,2026)
    
    title_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 22}
    plt.title(r'$\mathrm{CO_2}$ Concentration Data', fontdict=title_font)
    
    plt.plot( x_data, y_data, label='Data', marker= '.', color='b', markersize=3, linewidth=0)
    #plt.xlim( (x_min,x_max) )
    #plt.ylim( (y_min,y_max) )
    
    labels_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 20}
    plt.xlabel(r'Time (year)', fontdict=labels_font)
    plt.ylabel(r'$\mathrm{CO_2}$ concentration (ppm)', fontdict=labels_font)
    
#    # legend ->
#    plt.legend(loc=4)
#    # legend <-
#    
#    # Grid ->    
#    plt.grid(False)
#    # Grid <-  
    
#    # Ticks ->
    ax = plt.gca()
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
#    x_major = MultipleLocator(4); x_minor = MultipleLocator(2)
#   ax.xaxis.set_major_locator(x_major); 
    ax.xaxis.set_minor_locator(AutoMinorLocator())
#    y_major = MultipleLocator(20); y_minor = MultipleLocator(10)    
    ax.yaxis.set_major_locator(MultipleLocator(20)); 
    ax.yaxis.set_minor_locator(AutoMinorLocator())
#    #ax.xaxis.set_minor_locator(AutoMinorLocator(2))    
#    
    plt.tick_params(which='both', width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4)
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    #plt.tick_params(axis='both', which='minor', labelsize=8)
    plt.tight_layout() # for adjusting the bottom of x label
#    plt.yticks( range(370,470,20), (str(ss) for ss in range(370,470,20)), fontsize=20)
#    plt.xticks([2010,2014,2018,2022,2026], ['2010','2014','2018','2022','2026'],fontsize=20)
#    # Ticks <-
    plt.show()    
    # Plot <-    
    
def experiment2_fit_trend(trend_model='gp', optimize=False, optimize_file_prefix = '1', load_params=True, load_params_file_name=None):
    """
    This function fits the trend to the original raw data.
    
    This function must be run in order to find the parameters (hyperparameters) 
    of the trend.    
        
    Input:
    ------------------
    
    trend_model: string
        Which model is used for fitting
    
    optimize: bool
        Whether to perform trend parameters optimization.
    
    optimize_file_prefix: string
        When parameters optimized this is the prefix of the file with optimal parameters.
        Note, that 'trend_model' is added to the end or the file name anyway.
    
    load_params:
        Whether the trend parameters are loaded from the file
    
    load_params_file_name: string
        If parameters are loaded from the file, this tells the file name of the parameters.
    Output:
    ----------------------------
    
    The output is the plot of the trend(if no optimization happen) or
    optimization is done and parameters are saved in the file. File name see in the
    code.
    
    """
    
    (x_data,y_data) = load_data(detrend = False) # loads the raw data

    np.random.seed(234) # seed the random number generator just in case there are some initializations.
    
    if not load_params:
        variance_init = 40
        lengthcale_init = 400
        noise_var_init = 0.1
    else:
        #import pdb; pdb; pdb.set_trace()
        
        results_dict = io.loadmat( os.path.join(results_files_prefix, load_params_file_name) )
        
        params = results_dict['params'][0]
        variance_init = float(params[0])
        lengthcale_init = float(params[1])
        noise_var_init = float(params[2])
    
    if trend_model == 'gp':
        kernel = GPy.kern.RBF(1,variance=variance_init, lengthscale = lengthcale_init)
        
        model = GPy.models.GPRegression(x_data, y_data, kernel, noise_var=noise_var_init)
        
    elif trend_model == 'ss':
        if optimize:
            x_data = x_data[::10]; y_data = y_data[::10]
        kernel = GPy.kern.sde_RBF(1,variance=variance_init, lengthscale = lengthcale_init,
                                  balance= False, approx_order = 6 )
        kernel.lengthscale.constrain_bounded(100,500)
        model = GPy.models.StateSpace(x_data, y_data, kernel, noise_var=noise_var_init, balance=False, kalman_filter_type = 'svd')
        
    elif trend_model == 'sparse':
        if optimize:
            x_data = x_data[::40]; y_data = y_data[::40]
            
        import GPy.models.ss_sparse_model as ss_sparse_model
        kernel = GPy.kern.sde_RBF(1,variance=variance_init, lengthscale = lengthcale_init,
                                  balance= False, approx_order = 6 )
        kernel.lengthscale.constrain_bounded(100,500)
        
        model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=noise_var_init, 
                                                  balance=False, 
                                                  largest_cond_num=1e+20, regularization_type=2)
    #import pdb; pdb.set_trace()
    if optimize:
        model.optimize(messages=True)
        print(model)
        #import pdb; pdb.set_trace()
        
        result_dict = {}
        result_dict['params'] = model.param_array[:]
        save_file_name = optimize_file_prefix + '_' + trend_model
        print(save_file_name)
        io.savemat(os.path.join(results_files_prefix, save_file_name), result_dict)
    else:
        model.parameters_changed()
        
    print(model)
    if trend_model == 'gp':
        (y_pred,var_pred) = model.predict(x_data)
    elif trend_model == 'ss':
        (y_pred,var_pred) = model.predict(x_data, include_likelihood=True)
    elif trend_model == 'sparse':
        (y_pred,var_pred) = model.predict(x_data, include_likelihood=True, balance=False,
                                      largest_cond_num=1e+14, regularization_type=2)
        #(y_pred,var_pred) = model.predict(x_data)
    
    plt.figure(1)
     
    plt.plot( x_data, y_data, 'b.-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_data, y_pred, 'r-', label='Data',linewidth=1, markersize=5)
    plt.plot( x_data, y_pred+np.sqrt(var_pred), 'r--', label='Data',linewidth=1, markersize=5)
    plt.plot( x_data, y_pred-np.sqrt(var_pred), 'r--', label='Data',linewidth=1, markersize=5)
    
    plt.show() 
    


def experiment2_fit_state_space(hyperparameters='ss_optimal', detrend_file_name=None, other_params_file_name=None):
    """
    The model is StateSpace.
    Fit the model to the data and plot the result.
    
    Input:
    -------------------------------------------
    hyperparameters: string
        Which parameters (hyper-parameters) to use for fitting. 
        Two options: 'ss_optimal' and 'arno_optimal'. The first one is used here.
        The file name of the parameters is written in the text.
        
        The second one is older option for comparison.
    
    detrend_file_name: string
        Detrended file name.
    
    other_params_file_nameL string
        Other params file name.
        
    Output:
    --------------------------------------------
        Fitting plot
    
    """    
    import time
    
    # Data ->
    (x_data,y_data) = (x_data, y_data) = load_data(detrend = False)
    # Data <-
    
    #import pdb; pdb.set_trace()
    if hyperparameters == 'ss_optimal':
        
        #detrend_file_name = 'ex6_trend_fit_ss_2.mat'
        #other_params_file_name = '1_ex6_new_ss_0_scg__qp_bound_True'
        #other_params_file_name = '1_ex6_new_ss_2_scg__qp_bound_True'
        
        trend_params = io.loadmat( os.path.join(results_files_prefix, detrend_file_name) ); 
        trend_params = trend_params['params'][0]
        other_params = io.loadmat( os.path.join(results_files_prefix, other_params_file_name) )
        other_params = other_params['params'][0]
        
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
        var_1_trend = trend_params[0]
        ls_1_trend = trend_params[1]
        kernel4 =  GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend, balance=False, approx_order = 6)    
        
        # RBF kernel. Not used if the parameter 
        kernel = kernel1 * kernel2 + kernel3
        kernel.param_array[:] = other_params[:-1]
        kernel3.lengthscale = kernel3.lengthscale #*40 # !!! Redo optimizationwith different initial for this value        
        
        kernel = kernel + kernel4
        #kernel = kernel1 * kernel2 + kernel4
        kernel.fix() # fix all the parameters except the noise. Relevant when optimizing noise below.       
        noise_var = other_params[-1]       
      
        model = GPy.models.StateSpace( x_data, y_data, kernel=kernel, noise_var=noise_var, balance=False, kalman_filter_type = 'svd')        
        
        #model.optimize(messages=True)
        # The value below is obtained my uncommenting the line above and performing optimization.
        model.Gaussian_noise = 0.000358265664248      
        
    elif hyperparameters == 'arno_optimal':
        model_params = io.loadmat('/home/alex/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
        
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
        #kernel = kernel1 * kernel2 + kernel4
        
        model = GPy.models.StateSpace( x_data, y_data, kernel=kernel, noise_var=1.0, balance=False, kalman_filter_type = 'svd')
    
        model.param_array[:] = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale']/2,
                                    1.0, model_params['quasi_per_mat32_lengthscale'],  
                            model_params['mat32_inacc_magnSigma2'], model_params['mat32_inacc_lengthScale'], 
                model_params['rbf_magnSigma2'], model_params['rbf_lengthscale'], 
                                model_params['opt_noise'] ] )
        print(model.objective_function())
    print(model)
    # predict ->
    years_to_predict = 8
    step = np.mean( np.diff(x_data[:,0]) )
    
    x_new = x_data[-1,0] + np.arange( step, years_to_predict,  step ); x_new.shape = (x_new.shape[0],1)
    x_new = np.vstack( (x_data, x_new)) # combine train and test data
    
    t1 = time.time()
    ssm_mean, ssm_var = model.predict(x_new, include_likelihood=True)
    print('State Space prediction time {0:e} sec.'.format(time.time() - t1))
    # predict <-
    
    # Plot ->
    ssm_mean,ssm_std,x_data_denorm,y_data_denorm = denormalize_data(ssm_mean,np.sqrt(ssm_var))   
    x_new = x_new + 1974 # Put the right year
    
    plt.figure(1)
    (y_min,y_max) = (370,450)    
    (x_min,x_max) = (2010,2026)
    
    title_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 22}
    plt.title(r'State-space model prediction', fontdict=title_font)
    
    plt.plot( x_data_denorm, y_data_denorm, label='Data', marker= '*', color='0', markersize=5, linewidth=0)
    plt.plot( x_new, ssm_mean, color=(0,0,1), linestyle='-', label=r'$m$  (prediction mean)',linewidth=1, markersize=5)
    plt.plot( x_new, ssm_mean+2*ssm_std, color=(0,0,0.5), linestyle='--', label=r'$m\pm2\sigma$',linewidth=1)
    plt.plot( x_new, ssm_mean-2*ssm_std, color=(0,0,0.5), linestyle='--', label=None,linewidth=1)
    
    plt.plot( (x_data_denorm[-1], x_data_denorm[-1]),(380,440),  color=(0.2,0.2,0.2), linestyle='--', linewidth=1,
        label='Data/Prediction delimiter')
    plt.xlim( (x_min,x_max) )
    plt.ylim( (y_min,y_max) )
    
    labels_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 20}
    plt.xlabel(r'Time (year)', fontdict=labels_font)
    plt.ylabel(r'$\mathrm{CO_2}$ concentration (ppm)', fontdict=labels_font)
    plt.tight_layout() # for adjusting the bottom of x label
    # legend ->
    plt.legend(loc=4)
    # legend <-
    
    # Grid ->    
    plt.grid(False)
    # Grid <-  
    
    # Ticks ->
    ax = plt.gca()
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    x_major = MultipleLocator(4); x_minor = MultipleLocator(2)
    ax.xaxis.set_major_locator(x_major); ax.xaxis.set_minor_locator(x_minor)
    y_major = MultipleLocator(20); y_minor = MultipleLocator(10)    
    ax.yaxis.set_major_locator(y_major); ax.yaxis.set_minor_locator(y_minor)
    #ax.xaxis.set_minor_locator(AutoMinorLocator(2))    
    
    plt.tick_params(which='both', width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4)

    plt.yticks( range(370,470,20), (str(ss) for ss in range(370,470,20)), fontsize=20)
    plt.xticks([2010,2014,2018,2022,2026], ['2010','2014','2018','2022','2026'],fontsize=20)
    # Ticks <-
    plt.show()    
    # Plot <-    
    
def experiment2_fit_sparse_gp(hyperparameters='sparse_optimal',detrend_file_name=None, other_params_file_name=None):
    """
    The model is Sparse GP.
    Fit the model to the data and plot the result.
    
    Input:
    -------------------------------------------
    hyperparameters: string
        Which parameters (hyper-parameters) to use for fitting. 
        Two options: 'sparse_optimal' and 'arno_optimal'. The first one is used here.
        The file name of the parameters is written in the text.
        
        The second one is older option for comparison.
        
    detrend_file_name: string
        Detrended file name.
    
    other_params_file_nameL string
        Other params file name.
        
    Output:
    --------------------------------------------
        Fitting plot
    """
    import time
    
    import GPy.models.ss_sparse_model as ss_sparse_model
    
    # Data ->
    (x_data, y_data) = load_data(detrend = False)
    # Data <-
    
    if hyperparameters == 'sparse_optimal':
        
        #detrend_file_name = 'ex6_trend_fit_sparse_2.mat'
        #other_params_file_name = '1_ex6_new_sparse_0_scg__qp_bound_True_reg1e14'
        #other_params_file_name = '1_ex6_new_ss_2_scg__qp_bound_True'
        #other_params_file_name = '1_ex6_new_sparse_2_scg__qp_bound_True'
        
        trend_params = io.loadmat( os.path.join(results_files_prefix, detrend_file_name) ); 
        trend_params = trend_params['params'][0]
        other_params = io.loadmat( os.path.join(results_files_prefix, other_params_file_name) )
        other_params = other_params['params'][0]
    
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
        var_1_trend = trend_params[0]
        ls_1_trend = trend_params[1]
        kernel4 =  GPy.kern.sde_RBF(1, variance=var_1_trend, lengthscale=ls_1_trend, balance=False, 
                                    approx_order = 6)    
        
        
        kernel = kernel1 * kernel2 + kernel3        
        kernel.param_array[:] = other_params[:-1]
    
        kernel = kernel + kernel4
        
        kernel.fix() # fix all the parameters except the noise. Relevant when optimizing noise below. 
        noise_var = other_params[-1]
        model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=noise_var, balance=False, largest_cond_num=1e+14, regularization_type=2)
        #model.optimize(messages=True)
        # The value below is obtained my uncommenting the line above and performing optimization.
        model.Gaussian_noise = 0.000358573407244 # recompute
        
        print('Run sparse model')
        
    elif hyperparameters == 'arno_optimal':
        model_params = io.loadmat('/home/alex/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
        
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
        #kernel = kernel1 * kernel2 + kernel4
        
        model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=1.0, balance=False, largest_cond_num=1e+17, regularization_type=2)
        #import pdb; pdb.set_trace()
        model.param_array[:] = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale']/2,
                                    1.0, model_params['quasi_per_mat32_lengthscale'],  
                    model_params['mat32_inacc_magnSigma2'], model_params['mat32_inacc_lengthScale']*50, 
                          model_params['rbf_magnSigma2'], model_params['rbf_lengthscale'], 
                                    model_params['opt_noise'] ] )

    print(model)
    # predict ->
    years_to_predict = 8
    step = np.mean( np.diff(x_data[:,0]) )
    
    x_new = x_data[-1,0] + np.arange( step, years_to_predict,  step ); x_new.shape = (x_new.shape[0],1)
    x_new = np.vstack( (x_data, x_new)) # combine train and test data
    #import pdb; pdb.set_trace()
    t1 = time.time()
    sparse_mean, sparse_var = model.predict(x_new, include_likelihood=True, balance=False,
                                      largest_cond_num=1e+14, regularization_type=2)
    print('Sparse GP prediction time {0:e} sec.'.format(time.time() - t1))
    # predict <-
    
#     # Plot ->
#    sparse_mean,sparse_std,x_data_denorm,y_data_denorm = denormalize_data(sparse_mean,np.sqrt(sparse_var))   
#    x_new = x_new + 1974 # Put the right year
#    
#    plt.figure(2)
#    #plt.title('Electricity Consumption Data', fontsize=30)    
#    plt.plot( x_data_denorm, y_data_denorm, 'g-', label='Data',linewidth=1, markersize=5)
#    plt.plot( x_new, sparse_mean, 'b-', label='Mean',linewidth=1, markersize=5)
#    plt.plot( x_new, sparse_mean+2*sparse_std, 'r--', label='2*Std',linewidth=1, markersize=5)
#    plt.plot( x_new, sparse_mean-2*sparse_std, 'r--', label='2*Std',linewidth=1, markersize=5)
#    plt.xlim((2010,2026))
#    plt.ylim((370,450))
#    #plt.xlabel('Time (Hours)', fontsize=25)
#    #plt.ylabel('Normalized Value', fontsize=25)
#    #plt.legend(loc=2)
#    plt.show()    
#    # Plot <- 

    # Plot ->
    sparse_mean,sparse_std,x_data_denorm,y_data_denorm = denormalize_data(sparse_mean,np.sqrt(sparse_var))   
    x_new = x_new + 1974 # Put the right year
    
    plt.figure(2)
    (y_min,y_max) = (370,450)    
    (x_min,x_max) = (2010,2026)
    
    title_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 22}
    plt.title(r'Spin-GP prediction', fontdict=title_font)
    
    plt.plot( x_data_denorm, y_data_denorm, label='Data', marker= '*', color='0', markersize=5, linewidth=0)
    plt.plot( x_new, sparse_mean, color=(0,0,1), linestyle='-', label=r'$m$  (prediction mean)',linewidth=1, markersize=5)
    plt.plot( x_new, sparse_mean+2*sparse_std, color=(0,0,0.5), linestyle='--', label=r'$m\pm2\sigma$',linewidth=1)
    plt.plot( x_new, sparse_mean-2*sparse_std, color=(0,0,0.5), linestyle='--', label=None,linewidth=1)
    
    plt.plot( (x_data_denorm[-1], x_data_denorm[-1]),(380,440),  color=(0.2,0.2,0.2), linestyle='--', linewidth=1,
        label='Data/Prediction delimiter')
    plt.xlim( (x_min,x_max) )
    plt.ylim( (y_min,y_max) )
    
    labels_font = {'family': 'serif', 'color':  'k','weight': 'normal', 'size': 20}
    plt.xlabel(r'Time (year)', fontdict=labels_font)
    plt.ylabel(r'$\mathrm{CO_2}$ concentration (ppm)', fontdict=labels_font)
    plt.tight_layout() # for adjusting the bottom of x label
    # legend ->
    plt.legend(loc=4)
    # legend <-
    
    # Grid ->    
    plt.grid(False)
    # Grid <-  
    
    # Ticks ->
    ax = plt.gca()
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    x_major = MultipleLocator(4); x_minor = MultipleLocator(2)
    ax.xaxis.set_major_locator(x_major); ax.xaxis.set_minor_locator(x_minor)
    y_major = MultipleLocator(20); y_minor = MultipleLocator(10)    
    ax.yaxis.set_major_locator(y_major); ax.yaxis.set_minor_locator(y_minor)
    #ax.xaxis.set_minor_locator(AutoMinorLocator(2))    
    
    plt.tick_params(which='both', width=1)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4)

    plt.yticks( range(370,470,20), (str(ss) for ss in range(370,470,20)), fontsize=20)
    plt.xticks([2010,2014,2018,2022,2026], ['2010','2014','2018','2022','2026'],fontsize=20)
    # Ticks <-
    plt.show()    
    # Plot <-    



def experiment2_optimize_hyper_parameters(hyperparameters=1, p_model='sparse', optim = 'scg', 
                                          bound_qp_var=False, detrend_model = 'gp', detrend_file=None, save_file_name_prefix=None):
    """
    This function optimizes the hyperparameters of the model by gradient optimization.
    Note, that the trend is fitted separately and removed from the data
    before running optimization.
    
    Starting parameters are assigned a reasonable values.
    
    Input:
    ------------------------
    hyperparameters: None or int
        For optimization None must be fed. Number might be used if we want to
        load hyperparameters from a file and then start optimzation from them.
        Not used currently.
        
    p_model: string
        Defines a models, one of: 'gp', 'ss', 'sparse'
    
    optim: string
        Which optimization method is used, one of: 'scg', 'bfgs.'
        
    bound_qp_var: bool.
        Whether to bound quasi-periodic lengthscale. If not bounded some kernel
        may perform badly. Usually set to true.
    
    detrend_model: 'gp' or 'ss'
        Which model is used for detrending
        
    detrend_file: string
        Which parameters to take for detrending. Note that for detrending a GP
        model is used because for state-space and sparse the spamping is too
        dence wrt trend lengthscale, so they might perform badly.
    
    save_file_name_prefix: string
        Prefix of the resulting file name. The full file name is printed in the end
        and some other info is added to it.
        
    Output:
    --------------------------------
    Optimal hyperparameters are saved to a file.
    
    """ 
    #import pdb; pdb.set_trace()
    np.random.seed(237)
    
    import GPy.models.ss_sparse_model as ss_sparse_model
    
    # Data ->
    (x_data, y_data) = load_data(detrend = True, detrend_model = detrend_model, detrend_file=detrend_file)
    # Data <-
    #import pdb; pdb.set_trace()
    
    # Arno start hyper parameters ->
    if hyperparameters is not None:    
        model_params = io.loadmat('/home/alex/Programming/python/Sparse GP/solin-2014-supplement/arno_opt_params_to_python.mat')
        
#        tt = np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], 
#                        model_params['per_lengthscale']/2,
#                                    1.0, model_params['quasi_per_mat32_lengthscale'],  
#                                    model_params['mat32_inacc_magnSigma2'],
#                                     model_params['mat32_inacc_lengthScale'], 
#            model_params['rbf_magnSigma2'], model_params['rbf_lengthscale'], 
#                                model_params['opt_noise'] ] )
    
    
    # Periodic kernel
    var_1_per = 0.5 #1.0 # 5.0 
    ls_1_per = 1 / 2 # division by 2 is done because meanings in parameters are different
    period = 1    
    kernel1 = GPy.kern.sde_StdPeriodic(1, variance=var_1_per, lengthscale=ls_1_per, period=period, balance=False, approx_order = 6)
    if hyperparameters is not None:
        kernel1.param_array[:] = np.squeeze( np.array( [ model_params['per_magnSigma2'],  model_params['per_period'], model_params['per_lengthscale']/2 ] ) )
                                          # division by 2 is only for Arno start, otherwise use just pure param.                          
    kernel1.period.fix()
    if bound_qp_var:
        kernel1.lengthscale.constrain_bounded(0.4, 20000)

    # Quasiperiodic part
    var_1_per = 1.0 # does not change, fixed!
    ls_1_per = 50.0 #140.0
    kernel2 = GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
    if hyperparameters is not None:
        kernel2.param_array[:] = np.array( [ 1.0, model_params['quasi_per_mat32_lengthscale'] ] )
    kernel2.variance.fix()
    
    # Short term fluctuations kernel
    var_1_per = 0.1 #0.5
    ls_1_per = 50
    kernel3 =  GPy.kern.sde_Matern32(1, variance=var_1_per, lengthscale=ls_1_per)
    if hyperparameters is not None:
        kernel3.param_array[:] = np.array( [ model_params['mat32_inacc_magnSigma2'],
                                     model_params['mat32_inacc_lengthScale'] ] )
    
    # RBF kernel. Not used!
    var_1_per = 1
    ls_1_per = 100
    kernel4 =  GPy.kern.sde_RBF(1, variance=var_1_per, lengthscale=ls_1_per, balance=False, approx_order = 6)    
    if hyperparameters is not None:
        kernel4.param_array[:] = np.array( [ model_params['rbf_magnSigma2'], 
                                          model_params['rbf_lengthscale'] ] )
    noise_var = 0.1
    if hyperparameters is not None:
        noise_var = float(model_params['opt_noise'])
    
    kernel = kernel1 * kernel2 + kernel3# + kernel4
    # Arno start hyper parameters <-
    
    # Model ->
    if p_model == 'ss':            
        model = GPy.models.StateSpace( x_data, y_data, kernel=kernel, noise_var=noise_var, balance=False, kalman_filter_type = 'svd')
    elif p_model == 'sparse':
        # Regularization assume regularization type 2
        model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,kernel, noise_var=noise_var, balance=False, 
                                                  largest_cond_num=1e+14, regularization_type=2)
    else:
        raise ValueError("Wrong Parameter 1")
    # Model <-
    
    # Model ->
    #import pdb; pdb.set_trace()
#    model.kern.mul.Mat32.variance.fix()
#    model.kern.mul.std_periodic.period.fix()
    
    print(model)
    # Model <-
    
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
    hyp = str(hyperparameters) if hyperparameters is not None else '2'
    save_file_name = save_file_name_prefix + '_' +p_model+ '_' + hyp + '_' + optim + '_' + '_qp_bound_' +str(bound_qp_var)
    print(results_files_prefix)
    print(save_file_name)    
    io.savemat(os.path.join(results_files_prefix, save_file_name), result_dict)
    
    return model, oo


    # Plot <-
#def _helper_ss_sparse_compare_gradients(x_data, y_data, kernel, p_param_vector, p_step=1e-6):
#    """
#    
#    """
#    
#    k1 = kernel.copy()
#    ss_model = GPy.models.StateSpace( x_data, y_data, kernel=k1, noise_var=1.0, balance=False, kalman_filter_type = 'svd')
#    ss_model.param_array[:] = p_param_vector
#    ss_model.parameters_changed()
#    print(ss_model)
#    print(ss_model.checkgrad(verbose=True,step=p_step))
#    
#    k2 = kernel.copy()
#    import GPy.models.ss_sparse_model as ss_sparse_model
#    sparse_model = ss_sparse_model.SparcePrecisionGP(x_data,y_data,k2, noise_var=1.0, balance=False, largest_cond_num=1e+12, regularization_type=2) 
#    sparse_model.param_array[:] = p_param_vector
#    sparse_model.parameters_changed()
#    print(sparse_model)
#    print(sparse_model.checkgrad(verbose=True,step=p_step))
    
    
    
#def regularization_plot(reg_type = 2, reg=1e-6, zz = 1e-6):
#    """
#    Plot some regularization graphs. Completely independent of the rest
#    Calls are:
#    
#    #regularization_study(reg_type = 2, reg=1e-24)
#    #regularization_study(reg_type = 3, reg=1e-25, zz = 1e-13)
#    """
#    
#    if (reg_type == 2) or (reg_type == 3):
#        SV = 1e-14 * np.arange(1000,0,-1) # singular values 1e-11..1e-14
#        
#        inv_SV = 1/SV
#        
#        inv_SV_reg = SV/ (SV**2 + reg)
#        
#        
#        plt.figure(1)
#        #plt.title('Electricity Consumption Data', fontsize=30)    
#        plt.plot( SV, inv_SV, 'g.-', label='1/S',linewidth=1, markersize=5)
#        plt.plot( SV, inv_SV_reg, 'b.-', label='S/(S^2 + reg)',linewidth=1, markersize=5)
#        plt.plot( [np.sqrt(reg), np.sqrt(reg)], [np.min(inv_SV) , np.max(inv_SV)], 'r-',label='Comp. X Max' )
#        
#        plt.plot( [ SV[np.argmax(inv_SV_reg)], SV[np.argmax(inv_SV_reg)]], [np.min(inv_SV) , np.max(inv_SV)], 'm-',label='Empirical X Max' )
#        
#        plt.plot( [ SV[0], SV[-1]], [1/(2*np.sqrt(reg)) , 1/(2*np.sqrt(reg))], 'k-',label='Max new Value' )
#        
#        ax = plt.gca()
#        ax.invert_xaxis()
#        ax.ticklabel_format(axis='x', style='sci', scilimits=(-1,1))
#        plt.legend(loc=2)
#        
#    if (reg_type == 3):
#        
#        plt.plot( SV, 1/(SV + np.sqrt(reg)), 'c.-', label='1/(S+sqrt(reg))',linewidth=1, markersize=5)
#        #inv_SV_reg_2 = (SV + zz) / (SV**2 + reg)
#        inv_SV_reg_2 = (SV+np.sqrt(reg) + zz) / ((SV + np.sqrt(reg))**2 + reg)
#        #inv_SV_reg_2 = SV / (SV**2 + reg) +zz/(SV**3)
#    
#        plt.plot( SV, inv_SV_reg_2, 'k.-', label='(S+z)/(S^2 + reg)',linewidth=1, markersize=5)
#        
#    plt.show()
     
if __name__ == "__main__":

## 1) Plots the raw data
    #plot_raw_data()

## 2) Next we need to fit trends of regular GP (used for SpInGP) and State-Space by calling:
## experiment2_fit_trend(trend_model='gp', optimize=False, optimize_file_prefix = '1', load_params=True, load_params_file_name=None)

    #experiment2_fit_trend(trend_model='gp', optimize=True, optimize_file_prefix='ex2_trend_gp_new', load_params=False)
    ## Check the result:    
    #experiment2_fit_trend(trend_model='gp', optimize=False, load_params=True, load_params_file_name='ex2_trend_gp_new_gp')

    ## Now the same for the state-space models:
    #experiment2_fit_trend(trend_model='ss', optimize=True, optimize_file_prefix='ex2_trend_new', load_params=False)
    ## Check the result:    
    #experiment2_fit_trend(trend_model='ss', optimize=False, load_params=True, load_params_file_name='ex2_trend_new_ss')
    
## 3) After fitting the trend, we need to optimize other hyper-parameters. First-
##    for SpInGP and then for State-Space. File prefix are in the function body.
## Optimitzation cab be very long (tens of minutes)

    ## SpInGP: 
#    with warnings.catch_warnings(record=False):
#        warnings.simplefilter("once")
#        
#        experiment2_optimize_hyper_parameters(hyperparameters=None, p_model='sparse', optim = 'scg', 
#                                          bound_qp_var=True, detrend_model='gp', detrend_file='ex2_trend_gp_new_gp.mat',save_file_name_prefix='ex2_may') 
#        
                                          
    ## State-space model
#    with warnings.catch_warnings(record=False):
#        warnings.simplefilter("once")
#        experiment2_optimize_hyper_parameters(hyperparameters=None, p_model='ss', optim = 'scg', 
#                                          bound_qp_var=True, detrend_model='ss', detrend_file='ex6_trend_fit_ss_2',save_file_name_prefix='ex2') 
#                                          
        ## Result in: 'ex2_ss_2_scg__qp_bound_True'

# 4) After optimizing hyper parameters we can fit the SpInGP models and State-Space model:
#    File names are in the body of the function

    # Fit SpInGP:     
    #experiment2_fit_sparse_gp(hyperparameters='sparse_optimal', detrend_file_name='ex2_trend_gp', #other_params_file_name='1_ex6_new_sparse_2_scg__qp_bound_True') 
    #                          other_params_file_name='ex2_may_sparse_2_scg__qp_bound_True')
    
    # Fit State-Space
    #experiment2_fit_state_space(hyperparameters='ss_optimal', detrend_file_name='ex2_trend_ss.mat', other_params_file_name='ex2_ss_2_scg__qp_bound_True')






    # ex6_trend_fit_sparse_2   # 1_ex6_new_sparse_2_scg__qp_bound_True
    # experiment2_fit_sparse_gp(hyperparameters='sparse_optimal', detrend_file_name='ex2_trend_gp', other_params_file_name='1_ex6_new_sparse_2_scg__qp_bound_True')
    
    #experiment2_fit_state_space(hyperparameters='ss_optimal', detrend_file_name='ex2_trend_ss.mat', other_params_file_name='1_ex6_new_ss_2_scg__qp_bound_True')