#!/usr/bin/env python
# coding: utf-8

# In[1]:



# The MIT License (MIT)
# Modified work Copyright (c) [2016] [Kiran Gunnam]
# Add support for unknown being multiple data items instead of one data
# item
# Copyright (c) 2016 Markus Bergholz
# https://github.com/markuman/fastKNN
# Loop-Free KNN algorithm for GNU Octave and Matlab

# classified - result of KNN
# k
# nargin: the defined k
# nargout: information which k was taken (...when k was automatically determined!)
# idx - Index to map sorted distances dist to input dataset trained
# distance - default = 2
# distance == 2: Minkowski becomes equal Euclidean
# distance == 1: Minkowski becomes equal city block metric
# else: Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
# default with Euclidean distance and automagical determine of k

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import matplotlib.patches as mpatches
import scipy as sypy
from scipy import signal
from scipy import io
from scipy.stats import mode


def getDistance(x,y,p):  
    [my,ny]=np.shape(y)
    [mx,nx]=np.shape(x)
    idx_v=np.array(np.zeros((my,mx)),dtype=np.uint8)
    for i in range(my):
        z=y[i,::]
        v=sorted(np.sum(abs( (x[ ::,:-1:]-z) **p),1)**(1/p))
        idx= np.argsort(np.sum(abs( (x[ ::,:-1:]-z) **p),1)**(1/p))
    #         idx=sorted((np.sum(abs((x[ ::,:-1:]-z[np.ones(mx,1),:]**p),2))**(1/p)))
        idx_v[i,::]=idx
    return idx_v

def fastKNN(trained, unknown, **kwargs):
    
    [m,n]=np.shape(unknown)
    classified=np.array(np.zeros(m), dtype=np.uint8)
    
    l=len(kwargs)+2
    if(l<=3):
#       Minkowski Distance
#       for p == 2, Minkowski becomes equal Euclidean
#       for p == 1, Minkowski becomes equal city block metric
#       for p ~= 1 && p ~= 2 -> Minkowski https://en.wikipedia.org/wiki/Minkowski_distance
        distance=2
        
#   trained data has one more column as unknown, the class
    idx=getDistance(trained,unknown,distance)
    
    if(l<=2):
#       determine k value when no one is given
#       possible number of categories + 1
        k=np.size(np.unique(trained[::,-1])) +1   
    
    for i in range(m):
        tr=idx[i,0:k]
        for j in range(0,len(tr)):
#             trained_in[j]=trained[tr(j),:-1:]
            [mode_value, mode_count]=mode(trained[tr[j],-1])
        classified[i]=np.array(mode_value,dtype=np.uint8)        
    return [classified,k,idx]


# In[ ]:




