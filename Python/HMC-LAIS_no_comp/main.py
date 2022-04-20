# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:50:53 2021

@author: ErnestoAngelCurbeloB
"""

from lais_hamiltonian_multi import Lais_Ham
from norm_multi  import multi_normal_pdf
import numpy as np
from os.path import abspath,split
from time import time,strftime
import os
import pandas as  pd
from tabulate import tabulate

path,file = split(abspath(__file__))

cov_mat = np.array((4,3,3,4)).reshape((2,2))
cov = 4*np.eye(2)

# logaritmo de la densidad objetivo ===========================================
def logpdf_mix_multivariate(x,mu1=(0,0),cov_mat1=cov_mat,mu2=(-4,4),cov_mat2=cov_mat):
    s1 = 0.5*multi_normal_pdf(x,mu1,cov_mat1)
    s2 = 0.5*multi_normal_pdf(x,mu2,cov_mat2)
    
    return np.log(s1+s2)

print(path,file,sep="\n\n")


def simulacion(n_simulations,print_):
    folder_name = "simulation {}".format(strftime("%c")).replace(":","'")
    folder_location = os.path.join(path,folder_name)
    os.mkdir(folder_location)
    errores_medios = []
    epochs_ = []
    flags =[]
    dens = []
    parameters = []
    for simulation_i in range(n_simulations):
        if simulation_i%print_ ==0:
            print(f"Begins the simulation {simulation_i}")
        for param in params:
            
            step_length,path_length,epochs,den,flag=param
            chains = np.zeros((n_groups,epochs,dim))
            new_initial_points = np.random.uniform(-10,10,size=(n_groups,2))
            L = Lais_Ham(logpdf_mix_multivariate, cov_ham, epochs, step_length, path_length, initial_point,
                    cov_down,n_per_sample)
            for n in range(n_groups):
                L.upgrade_init_point(new_initial_points[n,:])
                # points.append(L.initial_point)
                chains[n,:,:] = L.upper_layer()
            if flag == "ham and lais": 
                new_samples = L.lower_layer(chains)
                new_samples,weights,Z = L.weighting(chains, new_samples,den)
            
                normalized_weights = weights/np.sum(weights)
                weights_flatted = normalized_weights.ravel(order="F")
                new_samples_flatted = new_samples.reshape((new_samples.shape[0]*new_samples.shape[1],dim))
            
                mu_est = 0
                second_moment_lais = 0
                
                for i in range(len(weights_flatted)):
                    mu_est = mu_est + weights_flatted[i]*new_samples_flatted[i]
                    second_moment_lais = second_moment_lais + weights_flatted[i]*new_samples_flatted[i]**2
                var_est_lais = second_moment_lais - mu_est**2
                aux = (new_samples_flatted - mu_est).prod(axis=1)
                cov_est_lais = (weights_flatted*aux).sum()
                
                    
                
                error_mu_lais = abs(real_mu - mu_est)
                error_cov_lais = abs(real_cov - cov_est_lais)
                error_var_lais = abs(real_var - var_est_lais)
                mean_error_lais = (error_mu_lais.sum() + error_var_lais.sum() + error_cov_lais)/5
                errores_medios.append(mean_error_lais)
                epochs_.append(epochs)
                flags.append("LAIS")
                dens.append(den)
                parameters.append((step_length,path_length,epochs))
                
                
            if den == "spatial":
                
                flatted_chains = chains.reshape((chains.shape[0]*chains.shape[1],chains.shape[2]))
                mu_est_ham = flatted_chains.mean(axis=0)
                second_moment_ham = (flatted_chains**2).mean()
                var_est_ham = flatted_chains.var(axis=0)
                cov_est_ham = (flatted_chains - mu_est_ham).prod(axis=1)
                cov_est_ham = cov_est_ham.mean()
                
                error_ham_mu = abs(real_mu - mu_est_ham)
                error_ham_var = abs(real_var - var_est_ham)
                error_ham_cov = abs(cov_est_ham - real_cov)
                mean_error_ham = (error_ham_mu.sum() + error_ham_var.sum() + error_ham_cov)/5
                
                errores_medios.append(mean_error_ham)
                epochs_.append(epochs)
                flags.append("HAM")
                dens.append("-")
                parameters.append((step_length,path_length,epochs))
        
        if simulation_i%print_ == 0:
            print(f"Finish the simulation {simulation_i}")

    
    b={"Errors":errores_medios,"Epochs":epochs_,"Denominator":dens,"Algorithm":flags,
       "Parameters":parameters}
    results = pd.DataFrame(b)
    
    address_to_save = os.path.join(folder_location,"{0} groups, {1} samples, {2} simulations, {3}.csv".format(n_groups,n_per_sample,n_simulations,time()))
    results.to_csv(address_to_save,index=False)
    
    data_to_save_txt = os.path.join(folder_location,"{0} groups, {1} samples, {2} simulations, {3}.txt".format(n_groups,n_per_sample,n_simulations,time()))
    file1 = open(data_to_save_txt,"w")
    file1.write(tabulate(b,headers=b.keys()))

    file1.close()
    
    
    for alg in ["HAM","LAIS"]:
        # step_length,path_length,epochs,den,alg = temp
        
        msk = results[results["Algorithm"] == alg]
        print(msk.columns)
        # a = results["Algorithm" == alg]
        
        if alg == "HAM":
            # c = results[results["Algorithm"=="HAM"]]
            for i in params_ham:
                d = msk[msk["Parameters"]==i]
                to_save_name = f"{alg},{i}.csv"
                address_to_save = os.path.join(folder_location,to_save_name)
                print(d)
                d["Errors"].to_csv(address_to_save,index=False)
        if alg == "LAIS":
            a_spatial = msk[msk["Denominator"] == "spatial"]
            a_temporal = msk[msk["Denominator"] == "temporal"]
            for i in params_lais:
                d_spatial = a_spatial[a_spatial["Parameters"]==i]
                d_temporal = a_temporal[a_temporal["Parameters"]==i]
                to_save_name_spatial = f"{alg},spatial,{i}.csv"
                address_to_save_spatial = os.path.join(folder_location,to_save_name_spatial)
                d_spatial["Errors"].to_csv(address_to_save_spatial)
                
                to_save_name_temporal = f"{alg},spatial,{i}.csv"
                address_to_save_temporal = os.path.join(folder_location,to_save_name_temporal)
                d_temporal["Errors"].to_csv(address_to_save_temporal)

    return

if __name__ == "__main__":
    
    """This code runs the simulations for HMC and HMC-LAIS.
    By running this code it creates a folder where will save the results 
    of the simulations. The parameters provided had to be:
    epochs, initial point, covariance of the momentum, amount of samples to 
    sample in the lower layer, covariance of the distributions in the lower
    layer and the denominator. For the sake of simulations the parameters are
    stored in the list 'parms', and the code iterates over this list."""
    
    # THIS ARE THE PARAMETERS THAT CAN BE CHANGED TO GET DIFFERNT REULTS
    epochs = 600 # HOW MANY ITERATIONS, 'T'.
    initial_point = (-4,2.5) # POINT TO START THE ITERATIONS.
    n_per_sample = 1 # AMOUNT OF SAMPLES TO SAMPLE IN THE LOWER LAYER.
    cov_ham = 2*np.eye(2) # COVARIANCE MATRIX OF THE MOMENTUM FOR HMC.
    cov_down = 2*np.eye(2) # COVARIANCE OF THE PROPOSALS IN THE LOWER LAYER.
    n_groups = 2 # AMOUNT OF CHAINS USED, 'N'.
    dim = len(initial_point) # DIMESION OF THE PROBLEM

    params = [(0.25, 1,2*epochs,"spatial","only ham"),(0.5,1,2*epochs,"spatial","only ham"),(1,3,2*epochs,"spatial","only ham"),(1,5,2*epochs,"spatial","only ham"),
              (0.25, 1,epochs,"spatial","ham and lais"),(0.5,1,epochs,"spatial","ham and lais"),(1,3,epochs,"spatial","ham and lais"),(1,5,epochs,"spatial","ham and lais"),
              (0.25, 1,epochs,"temporal","ham and lais"),(0.5,1,epochs,"temporal","ham and lais"),(1,3,epochs,"temporal","ham and lais"),(1,5,epochs,"temporal","ham and lais"),
             (0.25, 1,epochs,"all","ham and lais"),(0.5,1,epochs,"all","ham and lais"),(1,3,epochs,"all","ham and lais"),(1,5,epochs,"all","ham and lais") ] #step length/path length/epochs/den/algorithm
    
    params_ham = [(0.25, 1,2*epochs),(0.5,1,2*epochs),(1,3,2*epochs),(1,5,2*epochs),
                  (0.25, 1,epochs),(0.5,1,epochs),(1,3,epochs),(1,5,epochs)]
    params_lais = [(0.25, 1,epochs),(0.5,1,epochs),(1,3,epochs),(1,5,epochs)]
    
    # TRUE VALUES OF CUANTITIES CALCULATED
    real_mu = np.array((-2,2))
    real_cov = 0.5*(6 - 16) + 4
    real_var = np.array((8,8))
    
    # CALL TO THE MAIN FUNCTION: 'simulacion'
    simulacion(1,10)
