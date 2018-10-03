#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import matplotlib.patches as mpatches

def avgpool(img):
    m=np.size(img,0)
    n=np.size(img,1)
    pr= np.zeros((m//2,n//2))
    for r in range(0,m,2):
        for c in range(0,n,2):
            pr[(r+1)//2,(c+1)//2] = (img[r,c]+img[r+1,c]+img[r,c+1]+img[r+1,c+1])/4;
    return pr
    
    
def maxpool(img):
    m=np.size(img,0)
    n=np.size(img,1)
    pr= np.zeros((m//2,n//2))
    for r in range(0,m,2):
        for c in range(0,n,2):
            pr[(r+1)//2,(c+1)//2] = np.max(img[r,c]+img[r+1,c]+img[r,c+1]+img[r+1,c+1]);
    return pr


# In[ ]:




