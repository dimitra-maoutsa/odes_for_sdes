# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
from stochastic_models import climate_4d
import joblib
import importlib

MODEL = 'OU'
"""
MODEL can be: 
    - "OU"
    - "two_dim_multiwell"
    - "Lorentz"
    - "FHN_neuron"
                
"""

themodel = importlib.import_module("models."+MODEL)

dim = themodel.dim

sigma = 1
x0 = [-0,-0,5]
Ns = 4000 # number of stochastic particles
N = 4000  #number of particles
Ninf = 10000
#diffission coefficient/fucntion
gii = lambda x,t:np.array([1*sigma,0.01* sigma])#np.multiply(np.ones(dim),g)
xs = np.zeros((dim,Ninf))
for ii in range(dim):
    xs[ii] = np.random.normal(loc=x0[ii], scale=0.25,size=Ninf)