# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:50:53 2021

@author: ErnestoAngelCurbeloB
"""

from lais_hamiltonian_multi import Lais_Ham
from norm_multi  import multi_normal_pdf
import numpy as np
import matplotlib.pyplot as plt
from os.path import abspath,split
from time import time,strftime
import os
import pandas as  pd
from tabulate import tabulate
from sklearn.cluster import KMeans as kmeans

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
    clusters_used = []
    for simulation_i in range(n_simulations):
        
        if simulation_i%print_ ==0:
            
            print(f"Begins the simulation {simulation_i}")
            
        for param in params:

            step_length,path_length,epochs,den,flag,n_clusters=param

            chains = np.zeros((n_groups,epochs,dim))
            
            new_initial_points = np.random.uniform(-10,10,size=(n_groups,2))
            
            L = Lais_Ham(logpdf_mix_multivariate, cov_ham, epochs, step_length, path_length, initial_point,
                    cov_down,n_per_sample)
            
            for n in range(n_groups):
                
                L.upgrade_init_point(new_initial_points[n,:])
                # points.append(L.initial_point)
                
                chains[n,:,:] = L.upper_layer()
            
            if flag == "ham and lais": 

                # new_samples = L.lower_layer(chains)
                
                if den == "compression":

                    chains = chains.reshape((chains.shape[0]*chains.shape[1],chains.shape[2]))
                    # print(chains)
                    
                    NT_luca = chains.shape[0]

                    clt = kmeans(n_clusters)

                    clt.fit(chains)
                    
                    weights = np.empty(NT_luca)
                    
                    gaussian_weights= np.empty(n_clusters)
                    
                    gaussians_means = np.empty((n_clusters,chains.shape[1]))
                    
                    # gaussians_cov = np.empty((n_clusters,chains.shape[1],chains.shape[1]))
                    
                    # Q_mu = (np.matrix(chains - chains.mean(axis=0))).T*(np.matrix(chains - chains.mean(axis=0)))/(NT_luca)
                    
                    # centroids = clt.cluster_centers_
                    
                    # centroids_mu = centroids.mean(axis=0)
                    
                    # Q_c = (np.matrix(centroids - centroids_mu).T * np.matrix(centroids - centroids_mu))/n_clusters
                    
                    # Sigma = Q_mu - Q_c + cov_down
                    
                    Sigma = cov_down
                    
                    # weights = np.empty(NT_luca)
                    
                    new_samples = np.empty((NT_luca,chains.shape[1]))
                    
                    for c in range(n_clusters):
                        
                        cluster_c = chains[clt.labels_==c,:] 
                        
                        centroid_c = clt.cluster_centers_[c]
                        
                        gaussians_means[c,:] = centroid_c
                        
                        n_cluster_c = cluster_c.shape[0]
                        
                        gaussian_weights[c] = n_cluster_c/NT_luca
                        
                        
                        
                        cov_update = gaussian_weights[c]*(np.matrix(cluster_c - centroid_c)).T*(np.matrix(cluster_c - centroid_c))/(n_cluster_c)
                        
                        Sigma = Sigma + cov_update
                        
                        # gaussians_cov[c,:,:] = cov_update
                        
                        gaussian_weights[c] = n_cluster_c/NT_luca
        
                    for i in range(NT_luca):
                        
                        ind = np.random.choice(np.arange(n_clusters),p=gaussian_weights)
                      
                        samp = np.random.multivariate_normal(gaussians_means[ind,:],Sigma)

                        a = np.exp(L.logtarget(samp))
                        
                        b = 0
                        
                        for c in range(n_clusters):

                            b = b + gaussian_weights[c]*multi_normal_pdf(samp,gaussians_means[c,:],Sigma)
                    
                        new_samples[i,:] = samp    
                    
                        weights[i] = a/b
                    
                    normalized_weights = weights/np.sum(weights)
                    
                    weights_flatted = normalized_weights
                                        
                    new_samples_flatted = new_samples 
                    
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
                    
                    parameters.append((step_length,path_length))
                    
                    clusters_used.append(n_clusters)
                
                else:

                    new_samples,weights,Z = L.weighting(chains, new_samples,den,n_clusters)
                    
                
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
                    parameters.append((step_length,path_length))
                    clusters_used.append(n_clusters)
                
  

        if simulation_i%print_ == 0:
            
            print(f"Finish the simulation {simulation_i}")
   
    if len(clusters_used)>0:
        
        b={"Errors":errores_medios,"Epochs":epochs_,"Denominator":dens,"Algorithm":flags,
            "Parameters":parameters,"Clusters":clusters_used}
        
        results = pd.DataFrame(b)
    else:
        
        b={"Errors":errores_medios,"Epochs":epochs_,"Denominator":dens,"Algorithm":flags,
            "Parameters":parameters}
        
        results = pd.DataFrame(b)
    
    address_to_save = os.path.join(folder_location,"{0} groups, {1} samples, {2} simulations, {3}.csv".format(n_groups,n_per_sample,n_simulations,time()))
    results.to_csv(folder_name+"/sim.csv",index=False)
    
    data_to_save_txt = os.path.join(folder_location,"{0} groups, {1} samples, {2} simulations, {3}.txt".format(n_groups,n_per_sample,n_simulations,time()))
    file1 = open(folder_name+"/simu.txt","w")
    file1.write(tabulate(b,headers=b.keys()))
    file1.close()
    
    

    return

if __name__ == "__main__":
    """This code runs the simulations for HMC-LAIS with compression.
    By running this code it creates a folder where will save the results 
    of the simulations. The parameters provided had to be:
    epochs, initial point, covariance of the momentum, amount of samples to 
    sample in the lower layer, covariance of the distributions in the lower
    layer, the denominator. To perfor compression the 'denominator' has
    to 'compression' and then pass the number of clusters. For the sake of 
    simulations the parameters are stored in the list 'parms', and the code
    iterates over this list."""
    
    # THIS ARE THE PARAMETERS THAT CAN BE CHANGED TO GET DIFFERNT REULTS
    epochs = 300 # HOW MANY ITERATIONS, 'T'.
    initial_point = (-4,2.5) # POINT TO START THE ITERATIONS.
    n_per_sample = 1 # AMOUNT OF SAMPLES TO SAMPLE IN THE LOWER LAYER.
    cov_ham = 2*np.eye(2) # COVARIANCE MATRIX OF THE MOMENTUM FOR HMC.
    cov_down = 2*np.eye(2) # COVARIANCE OF THE PROPOSALS IN THE LOWER LAYER.
    # cov_down = np.array((4,3,3,10)).reshape((2,2))
    n_groups = 4 # AMOUNT OF CHAINS USED, 'N'.
    dim = len(initial_point) # DIMESION OF THE PROBLEM

    params = [
              (0.25, 1,epochs,"compression","ham and lais",3),
              (0.25, 1,epochs,"compression","ham and lais",21),
              (0.25, 1,epochs,"compression","ham and lais",50),
              (0.25, 1,epochs,"compression","ham and lais",200)] #step length/path length/epochs/den/algorithm/number of clusters
    
    params_ham = [(0.25, 1,2*epochs),(0.5,1,2*epochs),(1,3,2*epochs),(1,5,2*epochs),
                  (0.25, 1,epochs),(0.5,1,epochs),(1,3,epochs),(1,5,epochs)]
    
    params_lais = [(0.25, 1,epochs),(0.5,1,epochs),(1,3,epochs),(1,5,epochs)]

    # TRUE VALUES OF CUANTITIES CALCULATED   
    real_mu = np.array((-2,2))
    real_cov = 0.5*(6 - 16) + 4
    real_var = np.array((8,8))
 
    # CALL TO THE MAIN FUNCTION: 'simulacion'   
    simulacion(1,10)
