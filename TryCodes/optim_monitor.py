# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:04:28 2017

@author: agrigori

Model monitoring during optimization class
"""
import numpy as np
import scipy as sp

import GPy

class OptimMonitor(object):
    """
    This class is supposed to monitor two classes of models: GPRegression,
    Sparse GPs.
    
    Model must have correponding fields: e.g. ll, dll in order this class could monitor it
    
    """
    def __init__(self,):
        self.ll_data_term = []
        self.ll_det_term = []
        self.ll = []
        
        self.dll_data_term  = []
        self.dll_det_term = []
        self.dll = []
        
        self.mll_diff = [] # needed for the SparsePrecisionGP to compare differences
        self.d_mll_diff = []
        
    def clble(self, me, which=None):
        """
        This function is called each time the parameters change
        
        """
        import pdb; pdb.set_trace()
        
        if isinstance(me, GPy.models.gp_regression.GPRegression):
            self.ll.append( (-me.objective_function()) )
            self.dll.append( (-me.objective_function_gradients()).copy() ) # model.obj_grads
            
        elif isinstance(me, GPy.models.state_space_model.StateSpace):
            
            self.ll.append( (-me.objective_function()).copy() )
            self.dll.append( (-me.objective_function_gradients()).copy() ) # model.obj_grads
            
        elif isinstance(me, GPy.models.ss_sparse_model.SparcePrecisionGP):
            marginal_ll =me._marginal_ll
            d_marginal_ll = me._d_marginal_ll
            mll_data_fit_term = me._mll_data_fit_term
            mll_log_det = me._mll_log_det
            mll_data_fit_deriv = me._mll_data_fit_deriv
            mll_determ_deriv = me._mll_determ_deriv
        
            gpy_mll = (-me.objective_function())
            gpy_dmll = (-me.objective_function_gradients()).copy()
            self.ll.append( gpy_mll )
            self.dll.append( gpy_dmll ) # model.obj_grads
            
            self.ll_data_term.append(mll_data_fit_term)
            self.ll_det_term.append(mll_log_det)
            #self.ll.append()
            
            self.dll_data_term.append(mll_data_fit_deriv)
            self.dll_det_term.append(mll_determ_deriv)
            
            self.mll_diff.append( np.max(np.abs( marginal_ll - gpy_mll ) ) )
            self.d_mll_diff.append( np.max(np.abs( d_marginal_ll - gpy_dmll ) ) )
            #self.dll.append()