#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %%Kiran Gunnam

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import matplotlib.patches as mpatches
import scipy as sypy
import time
import os.path

from scipy import signal
from scipy import io
from numpy import ndarray

#from ipynb.fs.full.cnn
#import cnnload
#from ipynb.fs.full.avgpool
import avgpool
import cnn
#from ipynb.fs.full.avgpool
#import maxpool
#from ipynb.fs.full.cnn_training
import cnn_training
#from ipynb.fs.full.cnn_inference 
import cnn_inference

[trainlabels,trainimages,testlabels,testimages] = cnn.cnnload();

use_previous_training=0

maxtrain=2000; #maximum training images
iter= 10; # maximum iterations
eta=0.01; # learning rate
#
#  maxtrain=10000; #maximum training images
#  iter= 10; #maximum iterations
#  eta=0.01; # learning rate

#  maxtrain=60000; #maximum training images
#  iter= 30; #maximum iterations
#  eta=0.01; #learning rate

# select the pooling
# pool='maxpool';
pool= 'avgpool';


trained_parameter_file ='trained_parameters'+'_maxtrain'+str(maxtrain)+'_iter'+str(iter)+'_eta'+str(eta)+ pool+'.mat';

if(use_previous_training==0):
    tstart= time.time()
    cnn_training.cnn_training(trainlabels,trainimages,maxtrain,iter,eta,pool,trained_parameter_file);
    tfinish= time.time() -tstart
    if(os.path.isfile(trained_parameter_file)):
        print('training parameters are created');
else:
    if(os.path.isfile(trained_parameter_file)):
        print('using previously trained parameters');


tstart2= time.time()
[missimages, misslabels] = cnn_inference.cnn_inference(testlabels,testimages,pool,trained_parameter_file);
tfinish2= time.time()-tstart


# In[ ]:
