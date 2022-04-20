# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:17:24 2020

@author: ErnestoAngelCurbeloB
"""

def met_hast_multi(target,n_samples,initial_point,cov):
    """Function that samples from a target distribution
    using metropolis-hasting.
    In: 'target'--> density to sample from
        'n_samples'--> amount of samples to draw from the density
        'intial_point--> initial point of the chain
        'cov'--> covariance of the proposal density"""
    
    from numpy import zeros,array,random
    
    values = zeros((n_samples,len(initial_point))) #TO STORE THE SAMPLES
   
    z_all = zeros((n_samples,len(initial_point))) #TO STORE THE RECYCLED VALUES
   
    x_old = array(initial_point)
   
    evaluation_all = []
    
    for t in range(n_samples):
    
        z = array(random.multivariate_normal(x_old,cov))
        
        z_all[t,:] = z
        
        evaluation_all.append(multi_normal_pdf(z,x_old,cov))
        
        alpha = (target(z)*multi_normal_pdf(x_old,z,cov))/(target(x_old)*multi_normal_pdf(z,x_old,cov))
        
        alpha = min(1,float(alpha))
        
        assert alpha<=1
        
        r=random.random()
        
        if r<alpha:
        
            x_old = z
        
        values[t,:] = x_old

    return values,z_all,evaluation_all


def multi_normal_pdf(x,mu=None,cov=None):
    """Density function of the multivariate normal
    In: 'x'--> point to evaluate
        'mu'--> mean of the distribution
        'cov'--> matrix of variances and covariances"""
    
    from numpy.linalg import inv,det,cholesky
    
    from numpy import zeros,array,transpose,matmul,sqrt,pi,exp,eye
    
                  
    if type(mu)==type(None):
      mu=zeros(len(x))
    
    if type(cov)==type(None):
      cov = eye(len(x))
  
    k = len(x)
    
    x = array(x)#.reshape((len(x),1))
   
    mu = array(mu)#.reshape((len(mu),1))
   
    invers = inv(cov)
  
    if len(mu) == k and (len(mu),len(mu)) == cov.shape:
      
      try:
     
          flag = cholesky(cov)
      except:
        print("The covariance matrix is not positive definite")
        return
      
      d = det(cov)
      
      const = 1/( sqrt( d*(2*pi)**k ) )
      
      exponent = transpose(x-mu)


      exponent = matmul(exponent,invers)

      exponent = matmul(exponent,x-mu)

      exponent = exponent/(-2)
  
      return const*exp(exponent)

    else:

        raise "Las dimensiones no corresponden"

#------------------------------------------------------------------------------
      
def lais_lower_layer_several_multidimensional(chains,z_all,evaluation_all,cov,n_samples_per_chain):
    
    import numpy as np
    
    sig = cov
    
    samples = np.zeros((chains.shape[0],n_samples_per_chain*chains.shape[1],chains.shape[2]))
    
    samples.shape
    
    for j in range(chains.shape[0]):
        
        for i in range(chains.shape[1]):
        
            a = np.random.multivariate_normal(chains[j,i,:],cov,size=(n_samples_per_chain))
            
            samples[j,i*n_samples_per_chain:(i+1)*n_samples_per_chain,:] = a

    return samples

#------------------------------------------------------------------------------
def Phi(x,chains,cov,m):
    """Denominator of LAIS"""
    sum = 0
    m=int(m)
    N = chains.shape[0]
    for i in range(chains.shape[0]):
        sum += multi_normal_pdf(x,chains[i,m,:],cov)
    return sum/N

#-----------------------------------------------------------------------------
def weighting(target,chains,samples,cov):
    """Function that makes the weighting of the samples"""
    
    import numpy as np
    
    k = samples.shape[1]/chains.shape[1]
    
    weights = np.zeros((samples.shape[1],samples.shape[0]))
   
    for n in range(samples.shape[0]):

        for t in range(1,samples.shape[1]):

            p = Phi(samples[n,t,:],chains,cov,t//k-1)
        
            weights[t,n] = target(samples[n,t,:])/p

    Z = weights.sum()

    Z = Z/(samples.shape[1]*samples.shape[0])
    
    return samples,weights,Z

