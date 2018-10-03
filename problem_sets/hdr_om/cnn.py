#!/usr/bin/env python
# coding: utf-8

# In[1]:


# function [trainlabels,trainimages,testlabels,testimages] = cnnload()

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import matplotlib.patches as mpatches

def toint(b):
    x= b[0]*16777216 + b[1]*65536 + b[2]*256 + b[3];
    return(x)

def showimgs(img,lbl,h,w,fig):
    plt.figure
    for i in range (1,h*w):
        plt.subplot(h,w,i)
        plt.imshow(img[:,:,i])
        plt.title(num2str(lbl(i)))

def cnnload():
    trlblid = open('train-labels.idx1-ubyte','rb');
    trimgid = open('train-images.idx3-ubyte','rb');
    tslblid = open('t10k-labels.idx1-ubyte','rb');
    tsimgid = open('t10k-images.idx3-ubyte','rb');

    # read train labels

    np.fromstring(trlblid.read(4),dtype='uint8')
    numtrlbls= toint(np.fromstring(trlblid.read(4),dtype='uint8'))
    trainlabels = np.fromstring(trlblid.read(numtrlbls),dtype='uint8');

    # % read train data

    np.fromstring(trimgid.read(4),dtype='uint8')
    numtrimg=toint(np.fromstring(trimgid.read(4),dtype='uint8'))
    trimgh=toint(np.fromstring(trimgid.read(4),dtype='uint8'))
    trimgw=toint(np.fromstring(trimgid.read(4),dtype='uint8'))
    chunk_size=trimgh*trimgw*numtrimg
    chunk=np.fromstring(trimgid.read(trimgh*trimgw*numtrimg),dtype='uint8')
    trainimages=np.transpose(np.reshape(chunk,[numtrimg,trimgh,trimgw]),axes=[0,1,2])  # python index starts with 0

    # % read test labels

    np.fromstring(tslblid.read(4),dtype='uint8')
    numtslbls=toint(np.fromstring(tslblid.read(4),dtype='uint8'))
    testlabels=np.fromstring(tslblid.read(),dtype='uint8')


    # % read test data
    np.fromstring(tsimgid.read(4),dtype='uint8')
    numtsimg=toint(np.fromstring(tsimgid.read(4),dtype='uint8'))
    tsimgh=toint(np.fromstring(tsimgid.read(4),dtype='uint8'))
    tsimgw= toint(np.fromstring(tsimgid.read(4),dtype='uint8'))
    chunk_size=tsimgh*tsimgw*numtsimg
    chunk=np.fromstring(tsimgid.read(tsimgh*tsimgw*numtsimg),dtype='uint8')
    testimages=np.transpose(np.reshape(chunk,[numtsimg,tsimgh,tsimgw]),axes=[0,1,2])
    return trainlabels,trainimages,testlabels,testimages


# In[ ]:




