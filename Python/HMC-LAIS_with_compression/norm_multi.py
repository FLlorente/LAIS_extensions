# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 17:21:08 2021

@author: ErnestoAngelCurbeloB
"""

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
      #print((x-mu).shape)
      exponent = transpose(x-mu)
      #print(transpose(x-mu).shape)
      exponent = matmul(exponent,invers)
      exponent = matmul(exponent,x-mu)
      exponent = exponent/(-2)
  
  
      return const*exp(exponent)
    else:
      raise "Las dimensiones no corresponden"