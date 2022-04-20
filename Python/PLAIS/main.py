# create the function
def f(x,a,o):
  
    return np.exp(-a*x)*np.sin(o*x)

""" we define the functions: log_likelihood, prior, posterior y logposterior.
"""
def log_likelihood(y,data_with_error_observed,x_observed):
    a,b = y
    log_like = (-(1)/(2*0.1**2))*np.sum((data_with_error_observed - f(x_observed,a,b))**2)
    return log_like

def prior(y):
    a,b = y
    return 1*(0<=a)*(a<=10)*(0<=b)*(b<=2*np.pi)/(10*2*np.pi)


def prueba(n_groups,n_samples_mh,n_data):
    """This function calculates Z,E(X) and Var(X) using LAIS"""
    global x_chosen

    global data_with_error_observed_chosen
    
    chain_tensor = np.zeros((n_groups,n_samples_mh,2))
    z_all_tensor = np.zeros((n_groups,n_samples_mh,2))
    all_weights = []
  
    
    # this is the upper layer
    for i in range(n_groups):
      chosen = np.random.choice(len(data_with_error_observed),size=n_data,replace=False)
      x_chosen = x_observed[chosen]
      data_with_error_observed_chosen = data_with_error_observed[chosen]
      y01 = np.random.random()*10
      y02 = np.random.random()*2*np.pi
      y0 = [y01,y02]
      values,z_all,evaluation_all = met_hast_multi(exp_log_post_parcial,n_samples_mh,y0,cov)
      chain_tensor[i,:,:] = values
      z_all_tensor[i,:,:] = z_all
      x1,x2 = values[:,0],values[:,1]
      
      
    # this is the lower layer
    samples = lais_lower_layer_several_multidimensional(chain_tensor,z_all,evaluation_all,cov,1)

    samples_others,weights,Z = weighting(exp_log_post,chain_tensor,samples,cov)

    all_weights.append(weights)

    normalized_weights = weights/np.sum(weights)
    
    mu_est = 0
    sum2 = 0

    for i in range(n_samples_mh):
      for j in range(n_groups):
        mu_est = mu_est + normalized_weights[i,j]*samples[j,i,:]
        sum2 = sum2 + normalized_weights[i,j]*samples[j,i,:]**2
    var_est = sum2-mu_est**2
  
    return Z,mu_est,var_est

#===============================================================================
# these functions are used to make the aproximation by metropolis, because the ones are definde previously are with logarithms
def exp_log_post(y):
    return np.exp(log_post(y,data_with_error_observed=data_with_error_observed,x_observed=x_observed))

def exp_log_post_parcial(y):
    return np.exp(log_post_parcial(y,data_with_error_observed=data_with_error_observed_chosen,x_observed=x_chosen))


def simulations(n,n_groups,n_samples_mh,n_data):
    # "n_groups" es la cantidad de subposteriors a usar
    # n_groups = 10
    # n_samples_mh = 1000
    from time import time,strftime

    from tabulate import tabulate
    import os

    Zs = []
    mu_ests = []
    var_ests = []
    Z_error = []
    mu1_error = []
    mu2_error = []
    var1_error = []
    var2_error = []
    for i in range(n):
        t1 = time()
        Z,mu_est,var_est = prueba(n_groups,n_samples_mh,n_data)
        
        Z_error.append((Z-Z_true)**2)
        mu1_error.append((mu_est[0] - mu1_true)**2)
        mu2_error.append((mu_est[1] - mu2_true)**2)
        var1_error.append((var_est[0] - var1_true)**2)
        var2_error.append((var_est[1] - var2_true)**2)
    
        Zs.append(Z)
        mu_ests.append(mu_est)
        var_ests.append(var_est)
        t2 = time()

    b={"Values of Z":Zs,"Estimations of mu":mu_ests,"Estimations of the variance":var_ests,"Z errors":Z_error,"mu1 errors":mu1_error,"mu2 errors":mu2_error,"var1 errors":var1_error
       ,"var2 errors":var2_error}
    results = pd.DataFrame(b)
    address_to_save = os.path.join(folder_location,"{0} groups, {1} data, {2} samples, {3} simulations, {4}.csv".format(n_groups,n_data,n_samples_mh,n,time()))
    results.to_csv(address_to_save,index=False)
    
    data_to_save_txt = os.path.join(folder_location,"{0} groups, {1} data, {2} samples, {3} simulations, {4}.txt".format(n_groups,n_data,n_samples_mh,n,time()))
    file1 = open(data_to_save_txt,"w")
    file1.write(tabulate(b,headers=b.keys()))
    file1.write("\n \n MEANS OF ERRORS \n ")
    b1 = {"Z errors":[np.mean(Z_error)],"mu1 errors":[np.mean(mu1_error)],"mu2 errors":[np.mean(mu2_error)],"var1 errors":[np.mean(var1_error)]
       ,"var2 errors":[np.mean(var2_error)]}
    file1.write(tabulate(b1,headers=b1.keys()))
  
    file1.write("\n data in each subposterior: {} \n ".format(n_data))
    file1.write("amount of samples per chain in the lower layer: 1")
    file1.close()
    return

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import numpy as np
    from rlais_funcs import *
    import pandas as pd
    from time import time,strftime
    import os
    
    """I load the true values for the function. The address to the external files
    have to be changed"""
    fil = open("C:/Users/ErnestoAngelCurbeloB/Desktop/tfm_codes/códigos más claros/LAIS/true values.txt","r")
    line = fil.readlines()
    values = line[2].split("  ")
    Z,mu1,mu2,var1,var2 = values
    Z_true,mu1_true,mu2_true,var1_true,var2_true = float(Z),float(mu1),float(mu2),float(var1),float(var2)
    
    x_observed = np.loadtxt("C:/Users/ErnestoAngelCurbeloB/Desktop/tfm_codes/códigos más claros/LAIS/x_observed.txt")
    data_with_error_observed = np.loadtxt("C:/Users/ErnestoAngelCurbeloB/Desktop/tfm_codes/códigos más claros/LAIS/data_with_error_observed.txt")
    
    # CHOOSE THE DATA THAT WILL FORM THE PARTIAL POSTERIOR
    chosen = np.random.choice(len(data_with_error_observed),size=5,replace=False)
    x_chosen = x_observed[chosen]
    data_with_error_observed_chosen = data_with_error_observed[chosen]
    cov = 2 * np.eye(2)
    
    def log_post(y,data_with_error_observed=data_with_error_observed,x_observed=x_observed):
        a,b = y
        s = log_likelihood(y,data_with_error_observed,x_observed) + np.log(prior(y))
        return s
    def log_post_parcial(y,data_with_error_observed=data_with_error_observed_chosen,x_observed=x_chosen):
        a,b = y
        s = log_likelihood(y,data_with_error_observed,x_observed) + np.log(prior(y))
        return s
    def log_post_parcial(y,data_with_error_observed=data_with_error_observed_chosen,x_observed=x_chosen):
        a,b = y
        s = log_likelihood(y,data_with_error_observed,x_observed) + np.log(prior(y))
        return s


    # THIS IS THE ADDRESS TO SAVE THE RESULTS. IT HAS TO BE CHANGED
    folder_address = "C:/Users/ErnestoAngelCurbeloB/Desktop/tfm_codes/códigos más claros/LAIS/Simulaciones"
    folder_name = "simulation {}".format(strftime("%c")).replace(":","'")
    folder_location = os.path.join(folder_address,folder_name)
    
    os.mkdir(folder_location)
    
    # THIS ARE THE DIFFERENT PARAMETERS TESTED
    # params = [(1,1000,50),(2,500,25),(5,200,10),(10,100,5),(25,40,2),(50,20,1)]
    # params = [(1,1000,50),(2,500,50),(5,200,50),(10,100,50),(25,40,50),(50,20,50)] 
    # params = [(1,1000,10),(2,500,10),(5,200,10),(10,100,10),(25,40,10),(50,20,10)]  
    params = [(1,1000,5),(2,500,5),(5,200,5),(10,100,5),(25,40,5),(50,20,5)] #  (n_groups/n_samples_mh/n_data)
    n_simulations = 500 # HOW MANY SIMULATIONS TO PERFORM
    for i in params:
      n_groups,n_samples_mh,n_data = i
      simulations(n_simulations,n_groups,n_samples_mh,n_data)
    
