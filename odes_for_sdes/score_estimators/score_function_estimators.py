# -*- coding: utf-8 -*-

#Created on Sun Dec 12 03:35:29 2021

#@author: maout



### calculate score function from empirical distribution
### uses RBF kernel
import math
import numpy as np

from functools import reduce
from scipy.spatial.distance import cdist
import numba



__all__ = ["my_cdist", "score_function_multid_seperate", 
           "score_function_multid_seperate_all_dims", 
           "score_function_multid_seperate_old" ]



#%%
    

@numba.njit(parallel=True,fastmath=True)
def my_cdist(r,y, output,dist='euclidean'):
    """   
    Fast computation of pairwise distances between data points in r and y matrices.
    Stores the distances in the output array.
    Available distances: 'euclidean' and 'seucledian'
    
    Parameters
    ----------
    r : NxM array
        First set of N points of dimension M.
    y : N2xM array
        Second set of N2 points of dimension M.
    output : NxN2 array
        Placeholder for storing the output of the computed distances.
    dist : type of distance, optional
        Select 'euclidian' or 'sqeuclidian' for Euclidian or squared Euclidian
        distances. The default is 'euclidean'.

    Returns
    -------
    None. (The result is stored in place in the provided array "output").

    """
    N, M = r.shape
    N2, M2 = y.shape
    #assert( M == M2, 'The two inpus have different second dimention! Input should be N1xM and N2xM')
    if dist == 'euclidean':
        for i in numba.prange(N):
            for j in numba.prange(N2):
                tmp = 0.0
                for k in range(M):
                    tmp += (r[i, k] - y[j, k])**2            
                output[i,j] = math.sqrt(tmp)
    elif dist == 'sqeuclidean':
        for i in numba.prange(N):
            for j in numba.prange(N2):
                tmp = 0.0
                for k in range(M):
                    tmp += (r[i, k] - y[j, k])**2            
                output[i,j] = tmp   
    elif dist == 'l1':
        for i in numba.prange(N):
            for j in numba.prange(N2):
                tmp = 0.0
                for k in range(M):
                    tmp += (r[i, k] - y[j, k])**2          
                output[i,j] = math.sqrt(tmp)   
    return 0



def score_function_multid_seperate(X,Z,func_out=False, C=0.001,kern ='RBF',l=1,which=1,which_dim=1):
    """
    Sparse kernel based estimation of multidimensional logarithmic gradient of empirical density represented 
    by samples X across dimension "which_dim" only. 
    
    - When `funct_out == False`: computes grad-log at the sample points.
    - When `funct_out == True`: return a function for the grad log to be 
                                 employed for interpolation/estimation of
                                 the logarithmic gradient in the vicinity of the samples.
                               
    For estimation across all dimensions simultaneously see also
    
    See also
    ----------
    score_function_multid_seperate_all_dims 
    
    
    Parameters
    ----------
           X: N x dim array ,
              N samples from the density (N x dim), where dim>=2 the dimensionality of the system.
           Z: M x dim array,
              inducing points points (M x dim).
           func_out : Boolean, 
                      True returns function, if False return grad-log-p on data points.                    
           l: float or array-like,
               lengthscale of rbf kernel (scalar or vector of size dim).
           C: float,
              weighting constant (leave it at default value to avoid 
              unreasonable contraction of deterministic trajectories).         
           which: (depracated) ,
                 do not use.
           which_dim: int,
                      which gradient of log density we want to compute 
                      (starts from 1 for the 0-th dimension).
    Returns
    -------
            res1: array with logarithmic gadient of the density along the given dimension  N_s x 1 or function
                 that accepts as inputs 2dimensional arrays of dimension (K x dim), where K>=1.
    
    """
    
    if kern=='RBF':
        """
        #@numba.njit(parallel=True,fastmath=True)
        def Knumba(x,y,l,res,multil=False): #version of kernel in the numba form when the call already includes the output matrix
            if multil:                                        
                for ii in range(len(l)): 
                    tempi = np.zeros((x[:,ii].size, y[:,ii].size ), dtype=np.float64)
                    ##puts into tempi the cdist result                    
                    my_cdist(x[:,ii:ii+1], y[:,ii:ii+1],tempi,'sqeuclidean')
                    
                    res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                    
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ), dtype=np.float64)                
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                res = np.exp(-tempi/(2*l*l))
            #return 0
        """
        def K(x,y,l,multil=False):
            if multil:                         
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    #tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
                    ##puts into tempi the cdist result
                    #my_cdist(x[:,ii:ii+1], y[:,ii:ii+1],tempi,'sqeuclidean')
                    tempi = cdist(x[:,ii:ii+1], y[:,ii:ii+1],'sqeuclidean')
                    res = np.multiply(res, np.exp(-tempi/(2*l[ii]*l[ii])))                    
                return res
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ))                
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                return np.exp(-tempi/(2*l*l))
        
        def K1(x,y,l,multil=False):
            if multil:                
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                return res
            else:
                return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))
        
        #@njit
        def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            _,dim = x.shape            
            diffs = x[:,None]-y                         
            #redifs = np.zeros((1*N,N))
            ii = which_dim -1            
            if multil:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
            else:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
            
        """
        def grdy_K(x,y): # gradient with respect to the second argument
            _,dim = x.shape
            diffs = x[:,None]-y            
            #redifs = np.zeros((N,N))
            ii = which_dim -1              
            redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)         
            return -redifs
            
        #@njit        
        def ggrdxy_K(x,y):
            N,dim = Z.shape
            diffs = x[:,None]-y          
            
            redifs = np.zeros((N,N))
            for ii in range(which_dim-1,which_dim):  
                for jj in range(which_dim-1,which_dim):
                    redifs[ii, jj ] = np.multiply(np.multiply(diffs[:,:,ii],diffs[:,:,jj])+(l*l)*(ii==jj),K(x,y))/(l**4) 
            return -redifs
        """
            
            #############################################################################
    elif kern=='periodic': ###############################################################################################
      ###periodic kernel ###do not use yet!!!
        ## K(x,y) = exp(  -2 * sin^2( pi*| x-y  |/ (2*pi)  )   /l^2)
        
        ## Kx(x,y) = (K(x,y)* (x - y) cos(abs(x - y)/2) sin(abs(x - y)/2))/(l^2 abs(x - y))
        ## -(2 K(x,y) π (x - y) sin((2 π abs(x - y))/per))/(l^2 s abs(x - y))
      ##per = 2*np.pi ##period of the kernel
      #l = 0.5
      def K(x,y,l,multil=False):
        
        if multil:          
          
          res = np.ones((x.shape[0],y.shape[0]))                
          for ii in range(len(l)): 
              #tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
              ##puts into tempi the cdist result
              #my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi, 'l1')              
              #res = np.multiply(res, np.exp(- 2* (np.sin(tempi/ 2 )**2) /(l[ii]*l[ii])) )
              res = np.multiply(res, np.exp(- 2* (np.sin(cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'minkowski', p=1)/ 2 )**2) /(l[ii]*l[ii])) )
          return -res
        else:
            #tempi = np.zeros((x.shape[0], y.shape[0] ))
            ##puts into tempi the cdist result
            #my_cdist(x, y, tempi,'l1')
            #res = np.exp(-2* ( np.sin( tempi / 2 )**2 ) /(l*l) )
            res = np.exp(-2* ( np.sin( cdist(x, y,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            return res
        
      def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
          #N,dim = x.shape            
          diffs = x[:,None]-y   
          #print('diffs:',diffs)
          #redifs = np.zeros((1*N,N))
          ii = which_dim -1
          #print(ii)
          if multil:
              redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*K(x,y,l,True),diffs[:,:,ii] ),np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,np.cos( np.abs(diffs[:,:,ii])  / 2) ) , (l[ii]*l[ii]* np.abs(diffs[:,:,ii]))  ) 
          else:
              redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*diffs[:,:,ii],np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,K(x,y,l) ),np.cos( np.abs(diffs[:,:,ii]) / 2) ) ,(l*l* np.abs(diffs[:,:,ii])) )           
          return -redifs



    if isinstance(l, (list, tuple, np.ndarray)):
       ### for different lengthscales for each dimension 
       
       #numb-ed Kernel - uncomment this lines 
       #K_xz =  np.ones((X.shape[0],Z.shape[0]), dtype=np.float64) 
       #Knumba(X,Z,l,K_xz,multil=True)        
       #Ks =  np.ones((Z.shape[0],Z.shape[0]), dtype=np.float64) 
       #Knumba(Z,Z,l,Ks,multil=True) 
       K_xz = K(X,Z,l,multil=True) 
       Ks = K(Z,Z,l,multil=True)    
       multil = True
       
       Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
       A = K_xz.T @ K_xz           
       gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=True) #-
       
        
    else:
        multil = False
        
        K_xz = K(X,Z,l,multil=False) 
        
        Ks = K(Z,Z,l,multil=False)    
        
        Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
        A = K_xz.T @ K_xz    
        
        gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=False)
    sumgradx_K = np.sum(gradx_K ,axis=0)    
    if func_out==False: #if output wanted is evaluation at data points
        ### evaluatiion at data points
        res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K
    else:           
        #### for function output 
        if multil:             
            if kern=='RBF':      
                K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])
        
                
            elif kern=='periodic':
                K_sz = lambda x: np.multiply(np.exp(-2*(np.sin( cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), 'minkowski', p=2)/(l[0]*l[0])))),np.exp(-2*(np.sin( cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(l[1]*l[1])))))
            
        else:
            if kern=='RBF':
                K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
            elif kern=='periodic':
                K_sz = lambda x: np.exp(-2* ( np.sin( cdist(x, Z,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            

        res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K
    
    return res1





def score_function_multid_seperate_all_dims(X,Z,func_out=False, C=0.001,kern ='RBF',l=1):
    """
    Sparse kernel based estimation of multidimensional logarithmic gradient of empirical density represented 
    by samples X for all dimensions simultaneously. 
    
    - When `funct_out == False`: computes grad-log at the sample points.
    - When `funct_out == True`: return a function for the grad log to be employed for interpolation/estimation of grad log 
                               in the vicinity of the samples.
    
    Parameters
    -----------
            X: N x dim array,
               N samples from the density (N x dim), where dim>=2 the 
               dimensionality of the system.
            Z: M x dim array,
              inducing points points (M x dim).
            func_out : Boolean, 
                      True returns function, 
                      if False returns grad-log-p evaluated on samples X.                    
            l: float or array-like,
               lengthscale of rbf kernel (scalar or vector of size dim).
            C: float,
              weighting constant 
              (leave it at default value to avoid unreasonable contraction 
              of deterministic trajectories).
            kern: string,
                options:
                    - 'RBF': radial basis function/Gaussian kernel  
                    - 'periodic': periodic, not functional yet.           
           
    Returns
    -------
        res1: array with logarithmic gradient of the density  N_s x dim or function 
                 that accepts as inputs 2dimensional arrays of dimension (K x dim), where K>=1.
    
    """
    
    if kern=='RBF':
        """
        #@numba.njit(parallel=True,fastmath=True)
        def Knumba(x,y,l,res,multil=False): #version of kernel in the numba form when the call already includes the output matrix
            if multil:                                        
                for ii in range(len(l)): 
                    tempi = np.zeros((x[:,ii].size, y[:,ii].size ), dtype=np.float64)
                    ##puts into tempi the cdist result                    
                    my_cdist(x[:,ii:ii+1], y[:,ii:ii+1],tempi,'sqeuclidean')
                    
                    res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                    
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ), dtype=np.float64)                
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                res = np.exp(-tempi/(2*l*l))
            return 0
        """
        
        def K(x,y,l,multil=False):
            if multil:   
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
                    ##puts into tempi the cdist result
                    my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi,'sqeuclidean')
                    res = np.multiply(res,np.exp(-tempi/(2*l[ii]*l[ii])))                    
                    
                return res
            else:
                tempi = np.zeros((x.shape[0], y.shape[0] ))
                
                my_cdist(x, y,tempi,'sqeuclidean') #this sets into the array tempi the cdist result
                return np.exp(-tempi/(2*l*l))
            
        #@njit
        def grdx_K_all(x,y,l,multil=False): #gradient with respect to the 1st argument - only which_dim
            N,dim = x.shape    
            M,_ = y.shape
            diffs = x[:,None]-y                         
            redifs = np.zeros((1*N,M,dim))
            for ii in range(dim):          
            
                if multil:
                    redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
                else:
                    redifs[:,:,ii] = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
            
        
        def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            #_,dim = x.shape 
            #M,_ = y.shape
            diffs = x[:,None]-y                         
            #redifs = np.zeros((1*N,M))
            ii = which_dim -1            
            if multil:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])  
                
            else:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs
     
        
            #############################################################################
    elif kern=='periodic': ###############################################################################################
        ### DO NOT USE "periodic" yet!!!!!!!
      ###periodic kernel
        ## K(x,y) = exp(  -2 * sin^2( pi*| x-y  |/ (2*pi)  )   /l^2)
        
        ## Kx(x,y) = (K(x,y)* (x - y) cos(abs(x - y)/2) sin(abs(x - y)/2))/(l^2 abs(x - y))
        ## -(2 K(x,y) π (x - y) sin((2 π abs(x - y))/per))/(l^2 s abs(x - y))
      #per = 2*np.pi ##period of the kernel
      
      def K(x,y,l,multil=False):
        
        if multil:       
          res = np.ones((x.shape[0],y.shape[0]))                
          for ii in range(len(l)): 
              #tempi = np.zeros((x[:,ii].size, y[:,ii].size ))
              ##puts into tempi the cdist result
              #my_cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),tempi, 'l1')              
              #res = np.multiply(res, np.exp(- 2* (np.sin(tempi/ 2 )**2) /(l[ii]*l[ii])) )
              res = np.multiply(res, np.exp(- 2* (np.sin(cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'minkowski', p=1)/ 2 )**2) /(l[ii]*l[ii])) )
          return -res
        else:
            #tempi = np.zeros((x.shape[0], y.shape[0] ))
            ##puts into tempi the cdist result
            #my_cdist(x, y, tempi,'l1')
            #res = np.exp(-2* ( np.sin( tempi / 2 )**2 ) /(l*l) )
            res = np.exp(-2* ( np.sin( cdist(x, y,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            return res
        
      def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
          #N,dim = x.shape            
          diffs = x[:,None]-y             
          #redifs = np.zeros((1*N,N))
          ii = which_dim -1          
          if multil:
              redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*K(x,y,l,True),diffs[:,:,ii] ),np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,np.cos( np.abs(diffs[:,:,ii])  / 2) ) , (l[ii]*l[ii]* np.abs(diffs[:,:,ii]))  ) 
          else:
              redifs = np.divide( np.multiply( np.multiply( np.multiply( -2*diffs[:,:,ii],np.sin( np.abs(diffs[:,:,ii]) / 2) ) ,K(x,y,l) ),np.cos( np.abs(diffs[:,:,ii]) / 2) ) ,(l*l* np.abs(diffs[:,:,ii])) )           
          return -redifs

    dim = X.shape[1]

    if isinstance(l, (list, tuple, np.ndarray)):
       multil = True
       ### for different lengthscales for each dimension 
       #K_xz =  np.ones((X.shape[0],Z.shape[0]), dtype=np.float64) 
       #Knumba(X,Z,l,K_xz,multil=True) 
       
       #Ks =  np.ones((Z.shape[0],Z.shape[0]), dtype=np.float64) 
       #Knumba(Z,Z,l,Ks,multil=True) 
       K_xz = K(X,Z,l,multil=True) 
       Ks = K(Z,Z,l,multil=True)    
       
       #print(Z.shape)
       Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
       A = K_xz.T @ K_xz    
              
       gradx_K = -grdx_K_all(X,Z,l,multil=True) #-
       gradxK = np.zeros((X.shape[0],Z.shape[0],dim))
       for ii in range(dim):
           gradxK[:,:,ii] = -grdx_K(X,Z,l,multil=True,which_dim=ii+1)
       
       np.testing.assert_allclose(gradxK, gradx_K) 
    else:
        multil = False
        
        K_xz = K(X,Z,l,multil=False) 
        
        Ks = K(Z,Z,l,multil=False)    
        
        Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
        A = K_xz.T @ K_xz    
        
        gradx_K = -grdx_K_all(X,Z,l,multil=False)   #shape: (N,M,dim)
    sumgradx_K = np.sum(gradx_K ,axis=0) ##last axis will have the gradient for each dimension ### shape (M, dim)
    
    if func_out==False: #if output wanted is evaluation at data points
        
        # res1 = np.zeros((N, dim))    
        # ### evaluatiion at data points
        # for di in range(dim):
        #     res1[:,di] = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K[:,di]
        
        
        res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K        
        
        #res1 = np.einsum('ik,kj->ij', -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv, sumgradx_K)
        
        
    else:           
        #### for function output 
        if multil:      
            if kern=='RBF':      
                K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])        
                
            elif kern=='periodic':
                K_sz = lambda x: np.multiply(np.exp(-2*(np.sin( cdist(x[:,0].reshape(-1,1), Z[:,0].reshape(-1,1), 'minkowski', p=2)/(l[0]*l[0])))),np.exp(-2*(np.sin( cdist(x[:,1].reshape(-1,1), Z[:,1].reshape(-1,1),'sqeuclidean')/(l[1]*l[1])))))
            
        else:
            if kern=='RBF':
                K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))
            elif kern=='periodic':
                K_sz = lambda x: np.exp(-2* ( np.sin( cdist(x, Z,'minkowski', p=1) / 2 )**2 ) /(l*l) )
            

        res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K
        
            
        #np.testing.assert_allclose(res2, res1)
    
    return res1   ### shape out N x dim




def score_function_multid_seperate_old(X,Z,func_out=False, C=0.001,kern ='RBF',l=1,which=1,which_dim=1):
    
    """
    .. warning:: !!!This version computes distances with cdist from scipy. If numba is not available use this estimator.!!!!
    
    Sparse kernel based estimation of multidimensional logarithmic gradient of empirical density represented 
    by samples X across dimension "which_dim" only. 
    
    - When `funct_out == False`: computes grad-log at the sample points.
    - When `funct_out == True`: return a function for the grad log to be employed for interpolation/estimation of grad log 
                               in the vicinity of the samples.
                               
    
    
    Parameters
    -----------
            X: N samples from the density (N x dim), where dim>=2 the dimensionality of the system,
            Z: inducing points points (M x dim),
            func_out : Boolean, True returns function, if False return grad-log-p on data points,                    
            l: lengthscale of rbf kernel (scalar or vector of size dim),
            C: weighting constant (leave it at default value to avoid unreasonable contraction of deterministic trajectories)          
            which: return 1: grad log p(x) 
            which_dim: which gradient of log density we want to compute (starts from 1 for the 0-th dimension)
    Returns
    -------
        res1: array with density along the given dimension  N_s x 1 or function 
                 that accepts as inputs 2dimensional arrays of dimension (K x dim), where K>=1.
    
    
    For estimation across all dimensions simultaneously see also
    
    See also
    ---------
    score_function_multid_seperate_all_dims
    
    """
    if kern=='RBF':       
        
        def K(x,y,l,multil=False):
            if multil:                
                res = np.ones((x.shape[0],y.shape[0]))                
                for ii in range(len(l)): 
                    res = np.multiply(res,np.exp(-cdist(x[:,ii].reshape(-1,1), y[:,ii].reshape(-1,1),'sqeuclidean')/(2*l[ii]*l[ii])))
                return res
            else:
                return np.exp(-cdist(x, y,'sqeuclidean')/(2*l*l))            
        
        def grdx_K(x,y,l,which_dim=1,multil=False): #gradient with respect to the 1st argument - only which_dim
            #N,dim = x.shape            
            diffs = x[:,None]-y               
            #redifs = np.zeros((1*N,N))
            ii = which_dim -1            
            if multil:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l,True))/(l[ii]*l[ii])   
            else:
                redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)            
            return redifs            
        """
        def grdy_K(x,y): # gradient with respect to the second argument
            #N,dim = x.shape
            diffs = x[:,None]-y            
            #redifs = np.zeros((N,N))
            ii = which_dim -1              
            redifs = np.multiply(diffs[:,:,ii],K(x,y,l))/(l*l)         
            return -redifs            
                
        def ggrdxy_K(x,y):
            N,dim = Z.shape
            diffs = x[:,None]-y            
            redifs = np.zeros((N,N))
            for ii in range(which_dim-1,which_dim):  
                for jj in range(which_dim-1,which_dim):
                    redifs[ii, jj ] = np.multiply(np.multiply(diffs[:,:,ii],diffs[:,:,jj])+(l*l)*(ii==jj),K(x,y))/(l**4) 
            return -redifs            
        """
    if isinstance(l, (list, tuple, np.ndarray)):
       ### for different lengthscales for each dimension 
       K_xz = K(X,Z,l,multil=True) 
       Ks = K(Z,Z,l,multil=True)    
       multil = True ##just a boolean to keep track if l is scalar or vector       
       Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
       A = K_xz.T @ K_xz           
       gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=True)        
        
    else:
        multil = False
        K_xz = K(X,Z,l,multil=False) 
        Ks = K(Z,Z,l,multil=False)            
        Ksinv = np.linalg.inv(Ks+ 1e-3 * np.eye(Z.shape[0]))
        A = K_xz.T @ K_xz    
        gradx_K = -grdx_K(X,Z,l,which_dim=which_dim,multil=False)
    sumgradx_K = np.sum(gradx_K ,axis=0)
    if func_out==False: #For evaluation at data points!!!
        ### evaluatiion at data points
        res1 = -K_xz @ np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0]))@ Ksinv@sumgradx_K
    else:           
        #### For functional output!!!! 
        if multil:                            
            if kern=='RBF':      
                K_sz = lambda x: reduce(np.multiply, [ np.exp(-cdist(x[:,iii].reshape(-1,1), Z[:,iii].reshape(-1,1),'sqeuclidean')/(2*l[iii]*l[iii])) for iii in range(x.shape[1]) ])
        
        else:
            K_sz = lambda x: np.exp(-cdist(x, Z,'sqeuclidean')/(2*l*l))           

        res1 = lambda x: K_sz(x) @ ( -np.linalg.inv( C*np.eye(Z.shape[0], Z.shape[0]) + Ksinv @ A + 1e-3 * np.eye(Z.shape[0])) ) @ Ksinv@sumgradx_K

    
    return res1




#%%
    
