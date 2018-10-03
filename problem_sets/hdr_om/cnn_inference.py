#!/usr/bin/env python
# coding: utf-8

# In[114]:


# Kiran Gunnam
# separate into functions and have more configurability
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import matplotlib.patches as mpatches
import scipy as sypy
from scipy import signal
from scipy import io
from numpy import ndarray

# For testing function. 
# maxtrain=6; #maximum training images
# iter = 1; #maximum iterations
# eta=0.01; # learning rate

# n_fl=10;

# # %%select the pooling
# # pool='maxpool';
# pool= 'avgpool';
# trained_parameter_file = 'trained_parameters'+'_maxtrain'+str(maxtrain)+'_iter'+str(iter)+'_eta'+str(eta)+ pool+'.mat';
# [trainlabels, trainimages, testlabels, testimages] = cnnload()

from ipynb.fs.full.cnn import cnnload
from ipynb.fs.full.avgpool import avgpool
from ipynb.fs.full.avgpool import maxpool




def cnn_inference(testlabels,testimages,pool,trained_parameter_file):

    fn = 4; # number of kernels for layer 1
    ks = 5; # size of kernel

    [n,h,w]=np.shape(testimages)
    numtest=n;


    h1 = h-ks+1;
    w1 = w-ks+1;
    A1 = np.zeros((fn,h1,w1));

    h2 = h1//2;
    w2 = w1//2;
    I2 = np.zeros((fn,h2,w2));
    A2 = np.zeros((fn,h2,w2));

    A3 = np.zeros(10);


    tr_pr_fl=sypy.io.loadmat(trained_parameter_file)

    W1=tr_pr_fl['W1']
    W3=tr_pr_fl['W3']

    B1=tr_pr_fl['B1']
    B2=tr_pr_fl['B2']
    B3=tr_pr_fl['B3']

    S2=tr_pr_fl['S2']

    maxtrain=tr_pr_fl['maxtrain']

    it= tr_pr_fl['it']

    eta= tr_pr_fl['eta']

    err= tr_pr_fl['err']





    # normalize data to [-1,1] range
    nitest = (testimages / 255) * 2 - 1;
    miss = 0;

    missimages = np.zeros(numtest);
    misslabels = np.zeros(numtest);


    for im in range(0,numtest):
        for fm  in range (0,fn): 
            A1[fm,:,:] = sypy.signal.convolve2d(nitest[im,:,:],W1[fm, ::-1, ::-1], 'valid') + B1[:,fm]
        Z1 = np.tanh(A1);

    #     % Layer 2: max or average (both subsample) with scaling and bias
        for fm in range(0,fn):
            if(pool=='maxpool'):
             I2[fm,:,:] = maxpool(Z1[fm,:,:]);
            elif(pool=='avgpool'):
             I2[fm,:,:] = avgpool(Z1[fm,:,:]);
            A2[fm,:,:] = I2[fm,:,:] * S2[:,fm] + B2[:,fm];
        Z2 = np.tanh(A2);

    #     % Layer 3: fully connected
        for cl in range(0,10):
            A3[cl] = sypy.signal.convolve(Z2,W3[cl, ::-1, ::-1, ::-1],'valid') + B3[:,cl]
        Z3 = np.tanh(A3); # Final output

        pm = np.max(Z3);
        pl= np.argmax(Z3);
        if (pl != testlabels[im]+1):
            miss = miss + 1;
            missimages[miss] = im;
            misslabels[miss] = pl - 1;

    print(['Miss: ' + str(miss) +' out of ' +str(numtest)]);
    
    return missimages, misslabels


# In[ ]:




