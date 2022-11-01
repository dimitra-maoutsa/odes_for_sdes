# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 15:37:19 2018

@author: Dimi
"""


import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import pinv
from functools import reduce
from scipy.stats import gamma,norm,dweibull,tukeylambda,skewnorm
from matplotlib import pyplot as plt
from sklearn import preprocessing
from scipy.spatial.distance import cdist

### calculate score function from empirical distribution
### uses RBF kernel
### follows description of [Batz, Ruttor, Opper, 2016]

#Ktestsp = pdist2(xtrain',xsparse');
#Ktestsp= Ktestsp.^2/L^2;
#Ktestsp = exp(-Ktestsp);

def score_function(Z,f,D,T=1, C=0.25,kern ='RBF',p=4,l=1.,which=1,figs=False):
    """
    returns function psi(z)
    Input: Z: N observations
           f: function of known drift part
           C: weighting constant
           D: diffussion coefficient
           which: return 1: grad log p(x) , 2: log p(x), 3:both
    Output: psi: function
    
    """
### define kernel gradients
    if kern=='POLY':
        #### polynomial kernel
        p=4# power of polynomial
        
        def K(x,y):
            x = x.reshape((-1,1))
            y = y.reshape((-1,1))
            return np.power((1+ x@y.T),p)
            
        
        
        def grdx_K(x,y):
            x = x.reshape((-1,1))
            y = y.reshape((-1,1))
            return p*np.power(K(x,y)*y.T,p-1)
        
        def grdy_K(x,y):
            x = x.reshape((-1,1))
            y = y.reshape((-1,1))
            return p*x*np.power(K(x,y),p-1)
        
        
        def ggrdxy_K(x,y):
            x = x.reshape((-1,1))
            y = y.reshape((-1,1))
            return p*np.power(K(x,y),p-1)+ p*(p-1)*np.dot(x,y.T)*np.power(K(x,y)  ,p-2)
        
        def ggrdyy_K(x,y):
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            return p*(p-1)*x**2*np.power(K(x,y),p-2)
        
        def gggrdxyy_K(x,y):
            x = x.reshape((-1,1))
            y = y.reshape((-1,1))
            #return p*(p-1)*2*x*np.power((1+ np.dot(x,y)),p-2)+p*(p-1)*(p-2)*x*x*y* np.power((1+ np.dot(x,y)),p-3)
            return (p-1)*p*x* np.power(K(x,y),p-3)*(p*x@y.T+2)
        
    elif kern=='RBF':
        #l = 1. # lengthscale of RBF kernel

        def K(x,y):
            return np.exp(-(x-y.T)**2/(2*l*l))
            #return np.exp(np.linalg.norm(x-y.T, 2)**2)/(2*l*l) 
        def grdx_K(x,y):
            return -(1./(l*l))*(x-y.T)*K(x,y)
        
        def grdy_K(x,y):
            return (1./(l*l))*(x-y.T)*K(x,y)
        
        def ggrdxy_K(x,y):
            return (1./(l**4))*(-(x-y.T)**2+l**2)*K(x,y)
        def gggrdyxx_K(x,y):
            return (1./(l**6))*(x-y.T)*( (x-y.T)**2 -3*l*l ) * K(x,y)
        
        def gggrdxyy_K(x,y):
            return -(1./(l**6))*(x-y.T)*( (x-y.T)**2 -3*l*l ) * K(x,y)
                
        def ggrdyy_K(x,y):            
            return K(x,y)*((x-y.T)**2-l**2)/l**4
        
        def ggggrdxxyy_K(x,y):
            return K(x,y)*(1./l**6)*(3-6*(x-y)**2/l**2+(x-y)**4/l**4)
    
        
    
    elif kern=='RBFLin':
            l = 2. # lengthscale of RBF kernel
    
            def K(x,y):
                return np.exp(-(x-y.T)**2/(2*l*l))+(x-1)*(y.T-1)
                #return np.exp(np.linalg.norm(x-y.T, 2)**2)/(2*l*l) 
            def grdx_K(x,y):
                return -(1./(l*l))*(x-y.T)*np.exp(-(x-y.T)**2/(2*l*l))+(y.T-1)
            
            def grdy_K(x,y):
                return (1./(l*l))*(x-y.T)*np.exp(-(x-y.T)**2/(2*l*l))+(x-1)
            
            def ggrdxy_K(x,y):
                return (1./(l**4))*(-(x-y.T)**2+l**2)*np.exp(-(x-y.T)**2/(2*l*l))
            def gggrdyxx_K(x,y):
                return (1./(l**6))*(x-y.T)*( (x-y.T)**2 -3*l*l ) *np.exp(-(x-y.T)**2/(2*l*l))
            
            def gggrdxyy_K(x,y):
                return -(1./(l**6))*(x-y.T)*( (x-y.T)**2 -3*l*l ) * np.exp(-(x-y.T)**2/(2*l*l))
                    
            def ggrdyy_K(x,y):            
                return ((x-y.T)**2-l**2)/l**4 *np.exp(-(x-y.T)**2/(2*l*l))
            
            def ggggrdxxyy_K(x,y):
                return (1./l**6)*(3-6*(x-y)**2/l**2+(x-y)**4/l**4) *np.exp(-(x-y.T)**2/(2*l*l))
            
        
    N = Z.shape[0]
    Z = Z.reshape(-1,1)
    ### Evaluate Kernel Gradients at all N by N points
    #Ki = pairwise_kernels(Z, metric= K)
    
    ddKi = np.zeros((N,N))
    #dyKi = np.zeros((N,N))
    #dKi = np.zeros((N,N))#pairwise_kernels(Z, metric= grd_K)
    #k = lambda x: grdy_K(x,Z)
    #plt.figure(),
    #plt.plot(Z,k(Z[10]).T,'.')
    #G = np.zeros((N,N))#pairwise_kernels(Z, metric= ggrdxy_K)#,n_jobs=10)####G ddxyKi   
    ##ddyyKi = pairwise_kernels(Z, metric= ggrdyy_K) # 
    dddKi = np.zeros((N,N))
    Ki = np.zeros((N,N))
    #dddKi = pairwise_kernels(Z, metric= gggrd_K)
    #G2 = np.zeros((N,N))
    #precompute the drift function at data points
    #r = np.zeros(N)
    for i in range(N):
        #r[i] = 0#f(Z[i])
        #for j in range(N):
            #G[i,j] = ggrdxy_K(Z[i],Z[j])
            #dKi[i,j] = grdx_K(Z[i],Z[j])  
        ddKi[i,:] = ggrdxy_K(Z[i],Z)
        #print(ddKi.shape)
        ######################################Ki[i,:] = K(Z[i],Z)
        ########################################dddKi[i,:] = gggrdxyy_K(Z[i],Z)
        #print(gggrdxyy_K(Z[i],Z).shape)
        #dKi[i,:] = grdy_K(Z[i],Z)
    
    ######dddKi = gggrdxyy_K(Z,Z)
    #w,v = np.linalg.eig(ddKi)
    #plt.figure(),
    #plt.plot(np.real(w),np.imag(w),'r.')
    #####grad_y = -0.5*D*np.sum(dddKi, axis=1)#.T
    #ddyyKi = lambda x: 
    #print(ggrdyy_K(Z[1].reshape(-1,1),Z.T).shape)
    #####y = lambda x: 0.5*D*np.sum(ggrdyy_K(x,Z),axis=1)
    
    #def y(x):
    #    temp = []
    #    for i in range(x.size):
    #        temp.append(0.5*D*np.sum(ggrdyy_K(x[i],Z),axis=1))
    #    return np.array(temp)
           
    
    A = np.diagflat(np.ones(N))
    
    #g1 = np.linalg.inv( (np.eye(N,N)+ C* G @A)) 
    if T==0:
        T=N
    
    
    #gr = np.dot(G,r)    
    #g2 = C* ( gr + grad_y )  
    
    #g = - np.dot(g1, C* (  grad_y ))
    ##g =  g1@ (   grad_y )
    #g = np.dot(g1,dddKi)
    
    
    #psi = lambda x: -D*C*np.dot(k(x),g)-C*y #-C*(np.dot(k,r)+y)
    #psi =  -D*C*np.dot(k(Z).T,g)-C*y #-C*(np.dot(k,r)+y)
    #print(g.shape)
    #print(grad_y.shape)
    
    #psi = lambda x: - np.dot(np.dot( np.dot(C*k(x).T,A) , g1),C*grad_y) #-C*y(x)
    
    #psi = lambda x: - C*k(x)@A @ g1 @(C*grad_y)-C*y(x)
#    ofact =1#/( T*D/(N))
    #def psi(x):
    #    temp = []
    #    for i in range(x.size):            
            #temp.append(- C*k(x[i])@A @ g1 @(C*grad_y)-C*y(x[i]))
    #        temp.append( np.dot((np.dot(ddKi[i,:],g) - G[i,:]) ,(ofact*np.ones((N,1)) ))  )          
    #    return np.array(temp)
    #psi = psi(Z)
    pred = Z#np.linspace(-2,2,50)
    #temp2 = []
    #for i in range(Z.size):
    #    kxy = ggrdxy_K(Z[i], Z)
    #    kxyy = gggrdxyy_K(Z[i],Z)
    #    temp2.append( np.dot((np.dot(kxy,g) - kxyy) ,(ofact*np.ones((N,1)) ))        )
    #lfactor = N*D/T
    ofact = 1#T*D/(2*N)
    ###g1 = np.linalg.inv( 2*np.eye(N)+  ddKi ) 
    g1 = np.linalg.inv( 1*np.eye(N)+ C* ddKi ) 
    
    """
    plt.figure(figsize=(10,20)),
    plt.subplot(511)
    plt.imshow(ddKi, interpolation='none',origin='lower')
    plt.title(r'$\nabla \nabla K$')
    plt.colorbar()
    
    plt.subplot(512)
    plt.imshow(g1-np.diag(np.diag(g1)), interpolation='none',origin='lower')
    plt.title(r'$(2*I+\nabla \nabla K)^{-1}-diag(\cdot)$')
    plt.colorbar()
    
    plt.subplot(513)
    plt.imshow(g1, interpolation='none',origin='lower')
    plt.title(r'$(2*I+\nabla \nabla K)^{-1}$')
    plt.colorbar()
    
    plt.subplot(514)
    plt.imshow(np.log(g1-np.diag(np.diag(g1))), interpolation='none',origin='lower')
    plt.title(r'$log((2*I+\nabla \nabla K)^{-1}-diag(\cdot))$')
    plt.colorbar()
    
    plt.subplot(515)
    plt.imshow(np.log(-g1-np.diag(np.diag(-g1))), interpolation='none',origin='lower')
    plt.title(r'$log((2*I+\nabla \nabla K)^{-1}-diag(\cdot))$')
    plt.colorbar()
    """
    
    
    #print(np.linalg.inv(0.5*A))
    if figs:
        J = np.zeros((N,N))
    inx = [1,50,100,150,200-1]
    cols = ['k','m','b','r','g']
    ni = pred.size
    grad_y = np.zeros((ni,1))
    if which==1 or which==3:
#        temp3 = np.zeros(ni)
        
        for i in range(ni):
            
            kxyy = gggrdxyy_K(pred[i],Z)
#            temp3[i]=( 1*(C *kxy@g1@(C*dddKi)) -C* kxyy )@ (ofact*np.ones((N,1)) )  
            #print(( 0.5*(kxy@g1@(dddKi)) -0.5* kxyy ).shape)
            if figs:
                kxy = ggrdxy_K(pred[i], Z)
                J[i] = 0.5*(kxy@g1@(dddKi)) -0.5* kxyy
#            res = temp3
            grad_y[i] = 1* np.sum(D*kxyy) 
        temp3 = C*ggrdxy_K(Z, Z)@ g1 @(C*grad_y) - C*grad_y
        res=temp3
#        print(np.allclose(temp3b, temp3.reshape(-1,1)))
#        print(temp3)
#        print(temp3b)
        if figs==True:
            plt.figure()
            plt.plot(temp3,'.')
            plt.show()
            
            
            plt.figure(figsize=(10,10)),
            plt.subplot(411)
            plt.imshow(np.log(J), interpolation='none',origin='lower')
            #plt.title(r'$\nabla \nabla K$')
            plt.colorbar()
            plt.subplot(412)
            plt.imshow(np.log(-J), interpolation='none',origin='lower')
            #plt.title(r'$\nabla \nabla K$')
            plt.colorbar()
            plt.subplot(413)
            for en,entry in enumerate(inx):
                plt.plot(J[entry,:],'o' ,color=cols[en],label=entry)
            plt.plot([0,199],[0,0],'k--',alpha=0.5)
            plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
            plt.subplot(414)
            
            plt.plot(Z,'o')
            plt.plot([0,199],[0,0],'k--',alpha=0.5)
        #plt.title(r'$\nabla \nabla K$')
        #plt.colorbar()
        #psi2 = - 0.5( ofact *np.ones(N) ) * (  )
        ####psi = lambda x: -C*y(x) + np.dot(np.dot(k(x), np.linalg.inv(G+C*A*np.eye(N,N))),C*grad_y)
        #psi = lambda x: -C* C*A*np.dot(np.dot( k(x).T , g1),grad_y)
        ####psi = lambda x: np.dot(np.dot(k(x).T, 1/(C*A)+ G),C*grad_y)#-C*y(x)
        #print(k(Z).T.shape)
        #print(g1.shape)
        #print(np.dot( k(Z).T , g1).shape)
        #print(np.dot(np.dot( k(Z).T , g1),grad_y).shape)
        #print(y(Z).shape)
        ######psi =  -C* C*A*np.dot(np.dot( k(Z).T , g1),grad_y)-C*y(Z)
        
        #print('5') 
    if which==2 or which==3:
        temp5 = np.zeros(ni)
        #print(np.linalg.inv(C*A))
        #g2 = np.linalg.inv(ddKi + (np.eye(N)*2))
        for i in range(ni):
            ky = grdy_K(pred[i],Z)
            kyy = ggrdyy_K(pred[i],Z)
            ##temp5.append( 0.5*ky @ np.linalg.inv(ddKi + (np.eye(N)*2)) @ (( dddKi)@np.ones((N,1)))- 0.5*kyy@np.ones((N,1)))
            #temp5.append( ky @ np.linalg.inv(ddKi + (np.eye(N)*2)) @ ((C*(0.5*D) * dddKi)@np.ones((N,1)))- C*0.5*kyy@np.ones((N,1)))
            temp5[i]= (ky @ g1 @ ((C*(0.5*D) * dddKi))- C*0.5*kyy)@(ofact*np.ones((N,1))) 
        if which==2:
            res= temp5
        else:
            res = [res,temp5]
    
    return(res)#, w,v)

if __name__ == "__main__":
    cs = [ 0.001 ]
    ls = [7,8]
    for C in cs:
        for l in ls:
            g=np.sqrt(1)
            
            #gi=score_function(Y[:,0],'None',g**2,C)
            #plt.figure(),
            #plt.hist(gi,200)
            #plt.title(C)
            #plt.figure(),
            #plt.plot(sorted(gi),'.')
            #plt.show()
            #t_end=1
            #C= t_end/(Y.size*g**2)*10
            a = 4.
            r = norm(loc=a)#
            samples = np.sort(r.rvs(size=3000))
            #gpsi,trups=score_function(Y[:,0],'None',g**2,T=t_end,C=C,kern='RBF')
            psi,trups=score_function(samples,'None',g**2,T=samples.size,C=C,kern='RBF',which=3,l=l)
            #gi,psi,psipr, trup=score_function(Y[:,0],'None',g**2,t_end,C,kern='RBF')
            
            """
            inx = np.argsort(Y[:,0])
            plt.figure(),
            plt.plot(Y[:,0],f(Y[:,0],0),'k.'),
            plt.plot(Y[:,0][inx],psi[inx],'m.')
            plt.show()
            #plt.figure(),
            #plt.plot(Y[:,0][inx],(f(Y[:,0],0)[inx]-gi[inx])**2,'.'),        
            #plt.show()
            plt.figure(),
            plt.plot(Y[:,0],V(Y[:,0]),'k.'),
            plt.plot(Y[:,0][inx],trups[inx],'m.')
            plt.show()
            """
            
            """
            pred = Y[:,0]#np.linspace(-2,2,50)
            plt.figure(),
            plt.plot(pred,f(pred,0),'k.'),
            plt.plot(pred,gpsi,'m.')
            plt.show()
            #plt.figure(),
            #plt.plot(Y[:,0][inx],(f(Y[:,0],0)[inx]-gi[inx])**2,'.'),        
            #plt.show()
            plt.figure(),
            plt.plot(pred,V(pred),'k.'),
            plt.plot(pred,trups,'m.')
            plt.show()
            """
            #x = np.linspace(gamma.ppf(0.01, a),    gamma.ppf(0.99, a), 100)
            #plt.figure()
            #plt.plot(x[1:], np.diff((gamma.pdf(x, a))),'r-', lw=5, alpha=0.6, label='gradient gamma pdf')
            #plt.legend()
            print(C)
            
            x = np.linspace(gamma.ppf(0.01, a)-1,   gamma.ppf(0.99, a)+5, 100)
            #inx = np.argsort(samples)
            #pred = np.linspace(0,10,50)
            #plt.figure(),
            #plt.plot(samples,psi,'.',label='grad psi')
            #plt.legend()
            
            plt.figure(),
            plt.plot(samples,trups,'m.',label=' trupsi')
            plt.legend()
            plt.show()
            plt.figure(),
            plt.plot(samples,psi,'m.',label=' psi')
            plt.plot(samples,-(samples-a),'k')
            plt.legend()
            plt.show()
            #inx = np.argsort(Y[:,0])
            #plt.figure()
            #plt.plot(samples, np.log(gamma.pdf(samples, a)),'r.', lw=5, alpha=0.6, label='gamma pdf')
            ##plt.figure(),
            ##plt.plot(gi[inx],'.')
            #plt.figure()
            #plt.plot(samples,gpsi,'.')
            #plt.plot(x[1:], np.diff(np.log(gamma.pdf(x,a))),'r.', lw=5, alpha=0.6, label='grad log  pdf')
            #plt.legend()
            #plt.figure()
            #plt.plot(x, (np.log(gamma.pdf(x,a))),'k.', lw=5, alpha=0.6, label=' log  pdf')
            #plt.legend()
            #plt.show()
            #plt.figure(),
            #plt.plot(Y[:,0], psi(Y[:,0]),'r.')
            #plt.plot(Y[:,0], f(Y[:,0],0),'k.')
            
            """
            plt.figure(),
    
            #plt.plot(pred, np.array(psi)[:,0,0],'r.')
            #plt.plot(pred, f(pred,0),'k.')
            #plt.plot(Y[:,0],psi,'r.')
            plt.plot(pred,psir,'m.')
    
            plt.figure(),
            
            #plt.plot(pred, np.array(psi)[:,0,0],'r.')
            #plt.plot(pred, f(pred,0),'k.')
            #plt.plot(Y[:,0],psi,'r.')
            plt.plot(pred,psi,'m.')
            #plt.figure()
            #for i in range(1000):
            #    plt.plot(samples[i], )
            #plt.plot(samples,(psi(samples)),'.',label='psi')
            
            #plt.legend()
            
            
            #plt.figure()
            #plt.plot(samples, (dweibull.pdf(samples, a)),'r.', lw=5, alpha=0.5, label='dweibull pdf')
            #plt.legend()
            """
            
            """
            plt.figure()
            plt.plot(x, (np.log(norm.pdf(x,loc=a))),'r.', lw=5, alpha=0.6, label=' log gaussian pdf')
            plt.legend()
            plt.figure()
            plt.plot(x[1:], np.diff(np.log(norm.pdf(x,loc=a))),'r.', lw=5, alpha=0.6, label='grad log gaussian pdf')
            plt.legend()
            """
            """
            plt.figure()
            plt.plot(x, (np.log(skewnorm.pdf(x,a))),'r.', lw=5, alpha=0.6, label=' log  pdf')
            #plt.plot(x , 0.5*psi(x),'b.')
            plt.legend()
            plt.figure()
            plt.plot(x, psi,'k.', lw=5, alpha=0.6, label=' estimation log  pdf')
            plt.legend()
            
            
            
            plt.figure()
            xi = skewnorm.pdf(x,a)
            xi = preprocessing.Normalizer().fit_transform(xi.reshape(1, -1))
            #xi = preprocessing.scale(xi)
            plt.plot(x, xi.T,'r.', lw=5, alpha=0.6, label='  pdf')
            #plt.plot(x , 0.5*psi(x),'b.')
            #plt.legend()
            #plt.figure()
            xs = np.exp(psi(x))
            #xs = preprocessing.scale(xs)
            xs = preprocessing.Normalizer().fit_transform(xs.reshape(1, -1))
            plt.plot(x, xs.T,'k.', lw=5, alpha=0.6, label=' estimation   pdf')
            plt.legend()
            """        
        


"""
nx= np.arange(-2.,2.,0.1)
nxp = []
nxp1 = []
plt.figure(),
for entry in Y[inx,0]:
    #plt.plot(entry, psi(entry),'k.') 
    nxp.append(psi(entry)[0])
    nxp1.append(psi1(entry)[0])
plt.plot(np.array(nxp1)+np.array(nxp),'.')
plt.show()

l=1
def K(x,y):
    return np.exp(-(x.T-y)**2/(2*l*l))

def grdy_K(x,y):
    return (1./(l*l))*(x.T-y)*K(x,y)
    
    
"""