#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Kiran Gunnam
# separate into functions and have more configurability
# function [] = cnn_training(trainlabels,trainimages,maxtrain,iter,eta,pool,trained_parameter_file)


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import matplotlib.patches as mpatches
import scipy as sypy
from scipy import signal
from scipy import io

# Testing the function
# maxtrain=6; #maximum training images
# iter = 30; #maximum iterations
# eta=0.01; # learning rate
# n_fl=1;\
# # %%select the pooling
# # pool='maxpool';
# pool= 'avgpool';
# [trainlabels, trainimages, testlabels, testimages] = cnnload()

from ipynb.fs.full.cnn import cnnload
from ipynb.fs.full.avgpool import avgpool
from ipynb.fs.full.avgpool import maxpool

# function defintion here
def cnn_training(trainlabels, trainimages, maxtrain, iter,eta, pool, trained_parameter_file):
    fn = 4;  # number of kernels for layer 1
    ks = 5;  # size of kernel
    [n, h, w] = np.shape(trainimages);
    n = min(n, maxtrain);
    n_fl=10;

    # normalize data to [-1,1] range
    nitrain = (trainimages / 255) * 2 - 1;

    # train with backprop
    h1 = h - ks + 1;
    w1 = w - ks + 1;
    A1 = np.zeros((fn, h1, w1));

    h2 = int(h1 / 2);
    w2 = int(w1 / 2);
    I2 = np.zeros((fn,h2, w2));
    A2 = np.zeros((fn,h2, w2));
    A3 = np.zeros(10);

    # % kernels for layer 1
    W1 = np.random.randn(fn,ks, ks) * .01;
    B1 = np.ones(fn);

    # % scale parameter and bias for layer 2
    S2 = np.random.randn(1, fn) * .01;
    B2 = np.ones(fn);

    # % weights and bias parameters for fully-connected output layer
    W3 = np.random.randn(10,fn, h2, w2) * .01;
    B3 = np.ones(10);

    # % true outputs
    Y = np.eye(10) * 2 - 1;

    for it in range(0, iter):
        err = 0;
        for im in range(0, n):
            # ------------ FORWARD PROP ------------%
            # ------Layer 1: convolution with bias followed by tanh activation function
            for fm in range(0, fn):
                A1[fm, :, :,] = sypy.signal.convolve2d(nitrain[im, :, :], W1[fm, ::-1, ::-1], 'valid') + B1[fm];
            Z1 = np.tanh(A1)

            # ------Layer 2: max or average(both subsample) with scaling and bias

            for fm in range(0, fn):
                if pool == 'maxpool':
                    I2[fm, :, :] = maxpool(Z1[fm, :, :])
                elif pool == 'avgpool':
                    I2[fm, :, :] = avgpool(Z1[fm, :, :])
                A2[fm, :, :] = I2[fm, :, :] * S2[:,fm] + B2[fm]
            Z2 = np.tanh(A2)
            # ------Layer 3: fully connected

            for cl in range(0, n_fl):
                A3[cl] =sypy.signal.convolve(Z2, W3[cl, ::-1, ::-1, :: -1], 'valid') + B3[cl]

            Z3 = np.tanh(A3)
            err = err + 0.5*lin.norm(Z3.T - Y[:,trainlabels[im]],2)**2

            # ------------ BACK PROP ------------%
            # -------Compute error at output layer
            Del3 = (1 - Z3 ** 2) * (Z3.T - Y[:,trainlabels[im]]);

            #---Compute error at layer2
            Del2 = np.zeros(np.shape(Z2));
            for cl in range(0,10):
                Del2 = Del2 + Del3[cl] * W3[cl];

            Del2=Del2*(1- Z2**2)

            # Compute error at layer1
            Del1= np.zeros(np.shape(Z1))
            for fm in range(0,fn):
                Del1[fm,:,:]=(S2[:,fm]/4)*(1-Z1[fm,:,:]**2)
                for ih in range(0,h1):
                    for iw in range(0,w1):
                        Del1[fm,ih,iw]=Del1[fm,ih,iw]*Del2[fm,ih//2,iw//2]
                        
                        
            # Update bias at layer3
            DB3=Del3 # gradient w.r.t bias
            B3=B3 -eta*DB3

            # Update weights at layer 3
            for cl in range(0,10):
                DW3= DB3[cl] * Z2  #gradients w.r.t weights
                W3[3,:,:,:]=W3[cl,:,:,:] -eta*DW3
    
            # Update scale and bias parameters at layer 2            
            for fm in range(0,fn):
                DS2 = sypy.signal.convolve(Del2[fm,:,:],I2[fm, ::-1, ::-1],'valid')
                S2[:,fm]=S2[:,fm] -eta*DS2

                DB2=sum(sum(Del2[fm,:,:]))
                B2[fm]=B2[fm] -eta*DB2

           #Update kernel weights and bias parameters at layer 1
            for fm in range(0,fn):
                DW1 = sypy.signal.convolve(nitrain[im,:,:],Del1[fm, ::-1, ::-1],'valid')
                W1[fm,:,:]=W1[fm,:,:] -eta*DW1

                DB1=sum(sum(Del1[fm,:,:]))
                B1[fm]=B1[fm] -eta*DB1
        print(['Error: '+str(err)+' at iteration '+ str(it)])
    sypy.io.savemat(trained_parameter_file,{'W1':W1,'B1':B1,'S2':S2,'B2':B2,'W3':W3,'B3':B3,'maxtrain':maxtrain,'it':it,'eta':eta,'err':err})

