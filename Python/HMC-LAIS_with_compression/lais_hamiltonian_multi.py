# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:19:22 2021

@author: ErnestoAngelCurbeloB
"""
import numpy as np
from hamiltonian1 import Hamiltonian
from norm_multi import multi_normal_pdf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as kmeans
class Lais_Ham():
    """This class implements HMC-LAIS. It divides the algorithm in the two 
    modules: 'upper_layer' and 'lower_layer'. In the upper layer it calls the
    functions needed to sample using HMC from the module 'hamiltoninan1'"""
    def __init__(self,logtarget,M,n_samples_up,step_length, path_length, initial_point,cov_down,n_per_sample):
        self.logtarget = logtarget
        # self.n_samples = n_samples
        self.M = M
        self.epochs = n_samples_up
        self.step_length = step_length
        self.path_length = path_length
        self.initial_point = np.array(initial_point)
        self.cov_down = cov_down
        self.dim = len(initial_point)
        self.n_per_sample=n_per_sample
        
    def upper_layer(self,save_all = False):
        H = Hamiltonian(self.logtarget, self.M, self.epochs, self.step_length, self.path_length, self.initial_point)
        samples,all_steps = H.sampling()
        if save_all==False:
            return samples
        return samples,all_steps
    
    def lower_layer(self,samples):
        n_per_sample=self.n_per_sample

        if len(samples.shape)==3:
            n_chains,T,dim = samples.shape
            new_samples = np.zeros((n_chains,n_per_sample*T,dim))
            for n in range(n_chains):
                for t in range(T):
                    a = np.random.multivariate_normal(samples[n,t,:],self.cov_down,size=(n_per_sample))
                    #a  = z_all #z_all tiene que encajar con este
                    new_samples[n,t*n_per_sample:(t+1)*n_per_sample,:] = a
            
            return new_samples
            
        n_per_sample=self.n_per_sample
        new_samples = np.zeros((n_per_sample*samples.shape[0],samples.shape[1]))
        for j in range(samples.shape[0]):
            a = np.random.multivariate_normal(samples[j,:],self.cov_down,size=(n_per_sample))
            #a  = z_all #z_all tiene que encajar con este
            new_samples[j*n_per_sample:(j+1)*n_per_sample,:] = a
        return new_samples

    def Phi(self,x,chains,m,den,n=None,n_clusters=3,gaussian_weights=None,gaussians_means=None,gaussians_cov=None):
        """Denominators of LAIS"""
        cov = self.cov_down
        sum_ = 0
        m=int(m)
        if den == "temporal":
            if n == None:
                T = chains.shape[0]
                for t in range(chains.shape[0]):
                    sum_ += multi_normal_pdf(x,chains[t,:],cov)
            
            else:
                T = chains.shape[1]
                for t in range(T):
                    sum_ += multi_normal_pdf(x,chains[n,t,:],cov)
            
            return sum_/T
                    
        if den == "spatial":
            if n==None:
                pass
            
            else:
                sum_ = 0
                m=int(m)
                N = chains.shape[0]
                for n in range(chains.shape[0]):
                    sum_ += multi_normal_pdf(x,chains[n,m,:],cov)
            return sum_/N
        
        if den == "all":
            sum_ = 0
            if len(chains.shape) == 3:
                chains = chains.reshape((chains.shape[0]*chains.shape[1],chains.shape[2]))
                N = chains.shape[0]
            for t in range(chains.shape[0]):
                sum_ = sum_ + multi_normal_pdf(x,chains[t,:],cov)
            return sum_/N
        
        if den  == "compression":

            sum_ = 0

            for c in range(n_clusters):
                
                sum_ = sum_ + (gaussian_weights[c])*multi_normal_pdf(x,gaussians_means[c,:],gaussians_cov[c,:,:])
                
            return sum_
                
        
    def weighting(self,samples,new_samples,den,n_clusters=None,gaussian_weights=None,gaussians_means=None,gaussians_cov=None):
        
        if len(new_samples.shape)==2:
            k = new_samples.shape[0]/samples.shape[0]
            weights = np.zeros(new_samples.shape[0])
            
            for t in range(1,new_samples.shape[0]):
                p = self.Phi(new_samples[t,:],samples,t//k,den,n_clusters=n_clusters,gaussian_weights=gaussian_weights,gaussians_means=gaussians_means,gaussians_cov=gaussians_cov)
                weights[t] = np.exp(self.logtarget(new_samples[t,:]))/p
            Z = weights.sum()
            Z = Z/(new_samples.shape[0])
            
        if len(new_samples.shape) == 3:
            
            k = new_samples.shape[1]/samples.shape[1]
            weights = np.zeros((new_samples.shape[1],new_samples.shape[0]))
            
            for n in range(new_samples.shape[0]):
                for t in range(1,new_samples.shape[1]):
                    p = self.Phi(new_samples[n,t,:],samples,t//k,den,n,n_clusters,gaussian_weights,gaussians_means,gaussians_cov)
            
                    weights[t,n] = np.exp(self.logtarget(new_samples[n,t,:]))/p
            Z = weights.sum()
            Z = Z/(new_samples.shape[1]*new_samples.shape[0])
    
        return new_samples,weights,Z
    
    def upgrade_init_point(self,new_initial_point):
        self.initial_point = new_initial_point
    
    def upgrade_M(self,new_M):
        self.M = new_M
        
    def hist_(self,X,bins=50):
        x = X[:,0]
        y = X[:,1]
        plt.hist2d(x,y,bins=bins)
        plt.show()
        return
    
    def plot_(self,function,xlims,ylims,log=True):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        
        x0,x1 = xlims
        y0,y1 = ylims
        # return function((x0,y0))
        X = np.arange(x0,x1,0.1)
        Y = np.arange(y0,y1,0.1)
        if log==True:
            # Z = np.array([np.array([np.exp(function((x,y))) for x in X]) for y in Y])
            Z = np.array([[np.exp(function((x,y))) for x in X] for y in Y])
        else:
            Z = np.array([np.array([function((x,y)) for x in X]) for y in Y])
        
        X,Y = np.meshgrid(X,Y)
    
        surf=ax.contour(X,Y,Z)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        
        return
    
    def scatter_(self,X):
        x = X[:,0]
        y = X[:,1]
        plt.scatter(x,y)
        plt.show()
        return
