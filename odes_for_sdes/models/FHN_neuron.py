# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 01:19:29 2022

@author: maout
"""

import numpy as np
dim = 2
a = 0.1
b = 0.075
c = 0.1
I0 = 0.3
def f(x,t=0):
    return np.array([ x[0]*(a-x[0])*(x[0]-1)-x[1]+I0,-c*x[1]+b*x[0]])  # FHN neuron