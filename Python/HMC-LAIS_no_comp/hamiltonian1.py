# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:59:34 2021

@author: ErnestoAngelCurbeloB
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn


def gradiente(f,x,h=1e-6):
    try:
        x = np.array(x)
        n = len(x)
    except:
        n=1
    grad = np.zeros(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        grad[i] = (f(x+e*h) - f(x-e*h))/(2*h)
    return grad

class Hamiltonian():
    """This is the class of HMC. By calling the method 'samplin' we can get
    samples from the target density. It must be provieded with the logarithm
    of the density."""
    def __init__(self,logtarget,M,epochs,step_length,path_length,initial_point):
        self.logtarget = logtarget
        self.M = np.array(M)
        self.epochs = epochs
        self.step_length = step_length
        self.path_length = path_length
        self.initial_point = np.array(initial_point)
    
    def U(self,x):
        return -self.logtarget(x)
    
    def H(self,x,p,n=None):
        if n==1:

            b = p/self.M
            return self.U(x) + 0.5*p*b
        else:
            x= np.array(x)
            p = np.array(p)
            b = np.linalg.solve(self.M,p)
            return self.U(x) + 0.5*(p.dot(b))
    
    def sampling(self):
        
        steps = int(self.path_length/self.step_length)
        try:    
            n = len(self.initial_point)
        except TypeError:
            initial_point = (self.initial_point,)
            n = len(initial_point)

        
        # this is ran if dim == 1
        if n==1:
            initial_point = self.initial_point
            samples = np.zeros(self.epochs)
            samples[0] = initial_point
            # print(self.M)
            all_steps_q = np.zeros((self.epochs-1)*steps)
            all_steps_p = np.zeros((self.epochs-1)*steps)
            
            for e in range(1,self.epochs):
                
                q0 = samples[e-1]
                
                p0 = mvn.rvs(mean=np.zeros(n), cov=self.M)
            
                grad_u = gradiente(self.U,q0)
 
                p1 = np.copy(p0)

                q1 = np.copy(q0)

                for s in range(steps):

                    p1 = p1 - self.step_length*grad_u/2

                    b =  p1/self.M
                    
                    q1 = q1 + self.step_length*b
                    
                    all_steps_q[steps*(e-1)+s] = q1

                    grad_u = gradiente(self.U, q1)
                    
                    p1 = p1 - self.step_length*grad_u/2
                    all_steps_p[steps*(e-1)+s] = p1
                    
                p1 = -1*p1
                log_u = np.log(np.random.random())
                
                mh_test = -self.H(q1,p1,1) + self.H(q0,p0,1)
                
                if log_u<mh_test:
                    samples[e] = q1
                else:
                    samples[e] = q0
                
                if e%10 == 0:
                    print(f"Ended iteration {e}")
        # this is ran if dim > 1
        else:

            initial_point = np.array(self.initial_point)

            samples = np.zeros((self.epochs,n))

            samples[0,:] = initial_point

            all_steps_q = np.zeros((self.epochs*steps,n))
            
            for e in range(1,self.epochs):
                
                q0 = samples[e-1,:]

                p0 = mvn.rvs(mean=np.zeros(n), cov=self.M)

                grad_u = gradiente(self.U,q0)

                p1 = np.copy(p0)

                q1 = np.copy(q0)

                for s in range(steps):

                    p1 = p1 - self.step_length*grad_u/2

                    b = np.linalg.solve(self.M,p1)
                    
                    q1 = q1 + self.step_length*b

                    all_steps_q[steps*e+s,:] = q1

                    grad_u = gradiente(self.U, q1)
                    
                    p1 = p1 - self.step_length*grad_u/2
                
                p1 = -1*p1

                log_u = np.log(np.random.random())

                mh_test = -self.H(q1,p1) + self.H(q0,p0)
                
                if log_u<mh_test:

                    samples[e,:] = q1

                else:

                    samples[e,:] = q0
                
        return samples,all_steps_q #,all_steps_p
    
    def hist(self,arr):
        try:
            assert arr.shape[1]==2
        except:
            print("Points are not in the plane")
        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(1,1,1)
        ax.hist2d(arr[:,0],arr[:,1],bins=int(np.sqrt(arr.shape[0])))
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title("Histogram")
        plt.show()
