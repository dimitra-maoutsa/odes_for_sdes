#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:32:53 2020

@author: dimitra
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import coo_matrix
import joblib
from copy import deepcopy
folder_name = 'Lorenz63/'
def solve_Lorenz_63_Fokker(x0,t0,T, dt,rho=28,sigma=10,beta=8/3,sigma_x=10,sigma_y=10,sigma_z =10,initial_distr = False):
    """
    Computes the fokker planck solution of the Lorenz 63 system with parameters rho, simga, beta, and
    noise levels sigma_x, sigma_y, sigma_z
    x0: initial condition in samples --- initial_distr = False
        or initial distribution      --- initial_distr = True
    t0: starting time
    T: end time
    dt: step size
    initial_distr: determinines how the iniital condion will be treated
                    if True: x0 is considered as a distribution
                    if False: x0 is considered as samples from the initial distribution
    
    
    """
    #discretisation steps
    dx = 0.1
    dy = 0.2
    dz = 0.2
    
    #bounds for each dimension
    bnds_x = [-20,20]
    bnds_y = [-20,20]
    bnds_z = [0,10]
    
    #space grid
    x = np.arange(bnds_x[0], bnds_x[1] + dx, dx,dtype=np.float32) 
    y = np.arange(bnds_y[0], bnds_y[1] + dy, dy,dtype=np.float32) 
    z = np.arange(bnds_z[0], bnds_z[1] + dz, dz,dtype=np.float32) 
    print(z.size)
    #time_discretisation_grid
    timegrid = np.arange(t0,T+dt,dt,dtype=np.float32)
    
#    print(x)
#    print(y)
#    print(z)
    #array for the solution # size x.size,y.size,.z.size,timegrid
    nx = x.size 
    ny = y.size 
    nz = z.size 
    nt = timegrid.size
    Ntot = nx*ny*nz # total size of the resulting vector
    P = np.zeros((nx,ny,nz,nt),dtype=np.float32).reshape(-1,nt)  
    
    if initial_distr:
        #if x0 contains a distribution - from the previous run
        initial = x0
    else:
        #if initial contains samples from the initial condition
        #histogram to get probability density of initial points
        #append one extra block for the edges
        initial,_ = np.histogramdd(x0, bins = (np.append(x,bnds_x[1] + dx), np.append(y,bnds_y[1] + dy),np.append(z,bnds_z[1] + dz)),density=True)
        print(initial.shape)
    
    ## construct the matrix for the iteration
    #A = np.zeros((Ntot,Ntot),dtype=np.float32)
    
    a = (1+ dt *(sigma + 1 + beta)) - dt*2*sigma_x/dx**2 - dt*2*sigma_y/dy**2 - dt*2*sigma_z/dz**2
    row, col, data = [],[],[]
    for i in range(1,nx-1):
        
        for j in range(1,ny-1):
            
            for k in range(1,nz-1):
                
                #this is the index of the element i,j,k in the flattened array
                ii = i*(ny*nz)+j*nz+k
#                print('---')
#                print(ii)
#                print(i)
#                print(j)
#                print(k)
                ### set initial condition
                P[ii,0] = initial[i,j,k]
                
                #A[ii,ii] = a
                row.append(ii)
                col.append(ii)
                data.append(a)
                #### entries for partial derivative wrt z - b1 and b2 in my notes
                if ii < Ntot-1: 
                    #A[ii,ii+1] = dt*(beta*z[k]-x[i]*z[j])/(2*dz) + dt*sigma_z/dz**2
                    row.append(ii)
                    col.append(ii+1)                    
                    data.append(dt*(beta*z[k]-x[i]*y[j])/(2*dz) + dt*sigma_z/dz**2)
                if ii > 0:
                    #A[ii,ii-1] = -dt*(beta*z[k]-x[i]*z[j])/(2*dz) + dt*sigma_z/dz**2
                    row.append(ii)
                    col.append(ii-1)
                    data.append(-dt*(beta*z[k]-x[i]*y[j])/(2*dz) + dt*sigma_z/dz**2)
                
                
                #### entries for partial derivative wrt y
                if ii < Ntot - nz:
                    #A[ii,ii+nz] = dt*(y[j]-(x[i]*rho-x[i]*z[k]))/(2*dy) + dt*sigma_y/dy**2
                    row.append(ii)
                    col.append(ii+nz)
                    data.append(dt*(y[j]-(x[i]*rho-x[i]*z[k]))/(2*dy) + dt*sigma_y/dy**2)
                if ii > nz-1: 
                    #A[ii,ii-nz] = -dt*(y[j]-(x[i]*rho-x[i]*z[k]))/(2*dy) + dt*sigma_y/dy**2
                    row.append(ii)
                    col.append(ii-nz)
                    data.append(-dt*(y[j]-(x[i]*rho-x[i]*z[k]))/(2*dy) + dt*sigma_y/dy**2)
                
                
                ### entries for partial derivatives wrt x
                if ii < Ntot - nz*ny:
                    #A[ii,ii+nz*ny] = dt*sigma*(x[i]-y[j])/(2*dx) + dt*sigma_x/dx**2
                    row.append(ii)
                    col.append(ii+nz*ny)
                    data.append( dt*sigma*(x[i]-y[j])/(2*dx) + dt*sigma_x/dx**2)
                if ii > nz*ny - 1:
                    #A[ii,ii-nz*ny] = -dt*sigma*(x[i]-y[j])/(2*dx) + dt*sigma_x/dx**2
                    row.append(ii)
                    col.append(ii-nz*ny)
                    data.append( -dt*sigma*(x[i]-y[j])/(2*dx) + dt*sigma_x/dx**2)
    
    Asp = coo_matrix((data, (row,col)), shape=(Ntot, Ntot))
    for ti,tt in enumerate(timegrid):
        print(tt)
        if ti>0:
            
            # solve
            #P[:,ti] = A@P[:,ti-1]
            
            P[:,ti] = (Asp.dot(coo_matrix(P[:,ti-1].reshape(-1,1)))).toarray().reshape(-1,)
            
    
    del Asp
    P = P.reshape(nx,ny,nz,nt)
    
    return (P,(x,y,z,timegrid)) 
    
    
    
    
if __name__=="__main__":
    N=3000
    x0=np.random.normal(loc=[1,1,5], scale=0.25,size=(N,3))
    #total computation time
    T = 0.1
    #break it in chuncks of  t_in = 0.05 due to memory errors
    external_timegrid = np.arange(0, T+0.02, 0.02)
    
    #will split the computation in chunks according to the external timegrid
    for ti,tt in enumerate(external_timegrid[:-1]):
        print('>>>>>>>>>>>>>>>>>>>> %.3f to %.3f <<<<<<<<<<<<<<<<<<<'%(tt,external_timegrid[ti+1]))
        t0 = external_timegrid[ti]
        t1 = external_timegrid[ti+1]
        if ti==0:
            Ps,(nx,ny,nz,in_timegrid) = solve_Lorenz_63_Fokker(x0,t0,t1, dt=0.0001,rho=28,sigma=10,beta=8/3,sigma_x=10,sigma_y=10,sigma_z =10,initial_distr = False)
            Dict = {'Ps': Ps, 'nx':nx, 'ny':ny, 'nz':nz, 'in_timegrid':in_timegrid}
            joblib.dump(Dict,folder_name+ 'Lorenz_FP_rho_28_dt_0_0001_per_0_05_number_%d'%ti + '.gz', compress='gzip')  
        else:
            Ps,(nx,ny,nz,in_timegrid) = solve_Lorenz_63_Fokker(Ps[:,:,:,-1],t0,t1, dt=0.0001,rho=28,sigma=10,beta=8/3,sigma_x=10,sigma_y=10,sigma_z =10,initial_distr = True)
            Dict = {'Ps': Ps, 'in_timegrid':in_timegrid}
            joblib.dump(Dict,folder_name+ 'Lorenz_FP_rho_28_dt_0_0001_per_0_05_number_%d'%ti + '.gz', compress='gzip')  
            
    print('Fokker Planck calculation completed!!!')
    print('Combining the outputs')
    #####################
    #combine the outputs
    fine_timegrid = np.arange(0, T+0.0001, 0.0001)
    P_all = np.zeros((nx.size,ny.size,nz.size,int(fine_timegrid.size/10)))
    
    for ti,tt in enumerate(external_timegrid):
        
        Dict = joblib.load(folder_name+'Lorenz_FP_rho_28_dt_0_0001_per_0_05_number_%d'%ti + '.gz')
        Ps = Dict['Ps']
        in_timegrid = Dict['in_timegrid']
        P_all[:,:,:,ti:ti+in_timegrid.size] = deepcopy(Ps[:,:,:])
    print('Combined all results'  )
    
    print('Calculating the statistics....')
    
    #mean sum x*p(x)
    mean_x = np.zeros((fine_timegrid.size))
    mean_y = np.zeros((fine_timegrid.size))
    mean_z = np.zeros((fine_timegrid.size))
    
    std_x = np.zeros((fine_timegrid.size))
    std_y = np.zeros((fine_timegrid.size))
    std_z = np.zeros((fine_timegrid.size))
    
    skew_x = np.zeros((fine_timegrid.size))
    skew_y = np.zeros((fine_timegrid.size))
    skew_z = np.zeros((fine_timegrid.size))
    
    kurt_x = np.zeros((fine_timegrid.size))
    kurt_y = np.zeros((fine_timegrid.size))
    kurt_z = np.zeros((fine_timegrid.size))
    
    for ti,tt in enumerate(fine_timegrid):
        ###marginal statistics for x
        for ix,x in enumerate(nx):            
            mean_x[ti] += x*np.sum(P_all[ix,:,:,ti])        
        for ix,x in enumerate(nx):            
            std_x[ti] += np.sqrt((x-mean_x[ti])**2*np.sum(P_all[ix,:,:,ti]))
        for ix,x in enumerate(nx):            
            skew_x[ti] += ((x-mean_x[ti])/std_x[ti])**3*np.sum(P_all[ix,:,:,ti])
        for ix,x in enumerate(nx):            
            kurt_x[ti] += ((x-mean_x[ti])/std_x[ti])**4*np.sum(P_all[ix,:,:,ti])
            
        ###marginal statistics for y    
        for iy,y in enumerate(ny):            
            mean_y[ti] += y*np.sum(P_all[:,iy,:,ti])
        for iy,y in enumerate(ny):            
            std_y[ti] += np.sqrt((y-mean_y[ti])**2*np.sum(P_all[:,iy,:,ti]))
        for iy,y in enumerate(ny):            
            skew_y[ti] += ((y-mean_y[ti])/std_y[ti])**3*np.sum(P_all[:,iy,:,ti])
        for iy,y in enumerate(ny):            
            kurt_y[ti] += ((y-mean_y[ti])/std_y[ti])**4*np.sum(P_all[:,iy,:,ti])
            
        ###marginal statistics for z    
        for iz,z in enumerate(nz):            
            mean_z[ti] += z*np.sum(P_all[:,:,iz,ti])
        for iz,z in enumerate(ny):            
            std_z[ti] += np.sqrt((z-mean_z[ti])**2*np.sum(P_all[:,:,iz,ti]))
        for iz,z in enumerate(ny):            
            skew_z[ti] += ((z-mean_z[ti])/std_z[ti])**3*np.sum(P_all[:,:,iz,ti])
        for iz,z in enumerate(ny):            
            kurt_z[ti] += ((z-mean_z[ti])/std_z[ti])**4*np.sum(P_all[:,:,iz,ti])
                
                
        
        
    
    
    print('Saving the plots')
    for ti in np.arange(0,fine_timegrid.size,100):
        plt.figure(figsize=(16,10)),
        for i,ii in enumerate(np.arange(1,Ps.shape[2]-1,5)):
            plt.subplot(4,4,i+1)
            plt.imshow(P_all[:,:,ii,-2],origin='lower',extent = [nx[0] , nx[-1], ny[0] , ny[-1]])
            plt.xlabel('x')
            plt.ylabel('y')   
            plt.title( 'z= %.3f'%(nz[ii]) )
            plt.colorbar()
            
        plt.savefig(folder_name+'Lorenz63_fokker_planck_z_sections_time_%d'%ti+ '.png')
        plt.close()
            
        plt.figure(figsize=(16,10)),
        for i,ii in enumerate(np.arange(1,Ps.shape[1]-1,15)):
            plt.subplot(4,4,i+1)
            plt.imshow(P_all[:,ii,:,-2],origin='lower',extent = [nx[0] , nx[-1], nz[0] , nz[-1]])
            plt.xlabel('x')
            plt.ylabel('z') 
            plt.title( 'y= %.3f'%(ny[ii]) )
            plt.colorbar()
        plt.savefig(folder_name+'Lorenz63_fokker_planck_y_sections_time_%d'%ti+ '.png')
        plt.close()
            
        plt.figure(figsize=(16,10)),
        for i,ii in enumerate(np.arange(1,Ps.shape[0]-1,25)):
            plt.subplot(4,4,i+1)
            plt.imshow(P_all[ii,:,:,-2],origin='lower',extent = [ny[0] , ny[-1], nz[0] , nz[-1]])
            plt.xlabel('y')
            plt.ylabel('z')  
            plt.title( 'x= %.3f'%(nx[ii]) )
            plt.colorbar()
        plt.savefig(folder_name+'Lorenz63_fokker_planck_x_sections_time_%d'%ti+ '.png')
        plt.close()