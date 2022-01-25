# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 01:13:10 2022

@author: maout
"""
"two_dim_multiwell"
import numpy as np
## dimensionality of the system
dim = 2
#potential
a_1 = 1 #center/position of boundary
b_1 = 1 # distance between minima
a_2 = 1 #center/position of boundary
b_2 = 1 # distance between minima
Vmax = 4
V = lambda x,y: (Vmax/b_1**4)*((x-a_1)**2 - b_1**2)**2 + (Vmax/b_2**4)*((y-a_2)**2 - b_2**2)**2
#drift function (-grad V)
#f = lambda x,t: -np.array([4*Vmax*(x[0]-a_1)*((x[0]-a_1)**2-b_1**2)/b_1**4,4*Vmax*(x[1]-a_2)*((x[1]-a_2)**2-b_2**2)/b_2**4]).reshape(x.shape)

def f(x, t=0):
    return -np.array([4*Vmax*(x[0]-a_1)*((x[0]-a_1)**2-b_1**2)/b_1**4,4*Vmax*(x[1]-a_2)*((x[1]-a_2)**2-b_2**2)/b_2**4]).reshape(x.shape)
