# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 10:24:12 2017
@author: agrigori

In this file I test the possibility of expanding the information that is
stored on every step of optimization. This is useful for sparse models
where extra information needs to be stored on every iteration of optimization
e.g. mll parts and mll gradient parts.
"""
import numpy as np
import scipy as sp
import GPy

import optim_monitor

def generate_data(n_points,x_lower_value=0.0, x_upper_value=200.0):
    """
    Input:
    -----------
    
    n_points: number of data points.
    """

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
    
if __name__ == "__main__":

    np.random.seed(234) # seed the random number generator
    
    n_points = 100
    x_train_data, y_train_data = generate_data(n_points,0,1000)
    
    x_test_data, _ = generate_data(int(n_points/5),0,1000)
    #x_test_data, _ = generate_data(5,0,1000)
    
    # Kernel ->
    variance = np.random.uniform(0.1, 1.0) # 0.5
    lengthscale = np.random.uniform(0.2,10) #3.0
    noise_var = np.random.uniform( 0.01, 1) # 0.1
    
    kernel1 = GPy.kern.sde_Matern32(1,variance=variance, lengthscale=lengthscale)        
    kernel2 = GPy.kern.Matern32(1,variance=variance, lengthscale=lengthscale) 
    
    gp_reg = GPy.models.GPRegression(x_train_data, y_train_data, kernel2, noise_var=noise_var)
    
    sparse_reg = GPy.models.SparcePrecisionGP(x_train_data, y_train_data, kernel=kernel1, noise_var=1.0, balance=False, largest_cond_num=1e+16, 
         regularization_type=2)
         
#    def clbl(observ, which):
#         import pdb; pdb.set_trace()
#         print(observ.param_array)
#         #print(which)
#    
#    class tmp(object):
#        pass
#    tmp_obj = tmp()
#    
#    gp_reg.add_observer(tmp_obj, clbl)
    opt_mon1 = optim_monitor.OptimMonitor()
    opt_mon2 = optim_monitor.OptimMonitor()
    gp_reg.add_observer(opt_mon1, opt_mon1.clble)
    #gp_reg.optimize(optimizer='lbfgsb', messages=False)
    
    sparse_reg.add_observer(opt_mon2, opt_mon2.clble)
    sparse_reg.optimize(optimizer='lbfgsb', messages=True)
    