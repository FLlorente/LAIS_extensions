# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:30:00 2020

@author: ErnestoAngelCurbeloB
"""

import numpy as np

x_observed = np.loadtxt("C:/Users/ErnestoAngelCurbeloB/Desktop/trabajo con luca/rlais/x_observed.txt")
data_with_error_observed = np.loadtxt("C:/Users/ErnestoAngelCurbeloB/Desktop/trabajo con luca/rlais/data_with_error_observed.txt")

# create the function
def f(x,a,o):
    import numpy as np
    return np.exp(-a*x)*np.sin(o*x)

""" we define the functions: logverosimilitud, prior, posterior y logposterior.
The function log_post_partial can not be executed without excecute the previous cell 
"""
def log_likelihood(y,data_with_error_observed,x_observed):
    a,b = y
    log_like = (-(1)/(2*0.1**2))*np.sum((data_with_error_observed - f(x_observed,a,b))**2)
    return log_like

def prior(y):
    a,b = y
    return 1*(0<=a)*(a<=10)*(0<=b)*(b<=2*np.pi)/(10*2*np.pi)

#------------------------------------------------------------------------------
# def log_post(y,data_with_error_observed=data_with_error_observed,x_observed=x_observed):
#     a,b = y
#     s = log_likelihood(y,data_with_error_observed,x_observed) + np.log(prior(y))
#     return s
def log_post(y,data_with_error_observed,x_observed):
    a,b = y
    s = log_likelihood(y,data_with_error_observed,x_observed) + np.log(prior(y))
    return s
#------------------------------------------------------------------------------
# def log_post_parcial(y,data_with_error_observed=data_with_error_observed_chosen,x_observed=x_chosen):
#     a,b = y
#     s = log_likelihood(y,data_with_error_observed,x_observed) + np.log(prior(y))
#     return s

def log_post_parcial(y,data_with_error_observed,x_observed):
    a,b = y
    s = log_likelihood(y,data_with_error_observed,x_observed) + np.log(prior(y))
    return s
#===============================================================================
# these functions are used to make the aproximation by metropolis, because the ones are definde previously are with logarithms
def exp_log_post(y):
    return np.exp(log_post(y,data_with_error_observed=data_with_error_observed,x_observed=x_observed))

# def exp_log_post_parcial(y):
#     return np.exp(log_post_parcial(y,data_with_error_observed=data_with_error_observed_chosen,x_observed=x_chosen))

def exp_log_post_parcial(y):
    print(data_with_error_observed)
    return np.exp(log_post_parcial(y,data_with_error_observed,x_observed))



# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import numpy as np
#     a = 0.1
#     o = 2
#     x = np.arange(0,10,0.01)
#     fig = plt.figure(figsize=(10,8))
#     ax1 = fig.add_subplot(1,1,1)
#     ax1.plot(x,f(x,a,o))
#     plt.show()