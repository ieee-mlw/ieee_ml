#!/usr/bin/env python
# coding: utf-8

# In[1]:


# test_fastKNN.m
# Kiran Gunnam
# script based on README
# Dataset taken from http://www.jiaaro.com/KNN-for-humans/

#   -------------------------------------------------------
#     |  weight (g)  |  color  |  # seeds  ||  Type of fruit  |
#     |==============|=========|===========||=================|
#     |  303         |  3      |  1        ||  Banana         |
#     |  370         |  1      |  2        ||  Apple          |
#     |  298         |  3      |  1        ||  Banana         |
#     |  277         |  3      |  1        ||  Banana         |
#     |  377         |  4      |  2        ||  Apple          |
#     |  299         |  3      |  1        ||  Banana         |
#     |  382         |  1      |  2        ||  Apple          |
#     |  374         |  4      |  6        ||  Apple          |
#     |  303         |  4      |  1        ||  Banana         |
#     |  309         |  3      |  1        ||  Banana         |
#     |  359         |  1      |  2        ||  Apple          |
#     |  366         |  1      |  4        ||  Apple          |
#     |  311         |  3      |  1        ||  Banana         |
#     |  302         |  3      |  1        ||  Banana         |
#     |  373         |  4      |  4        ||  Apple          |
#     |  305         |  3      |  1        ||  Banana         |
#     |  371         |  3      |  6        ||  Apple          |
#      -------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import matplotlib.patches as mpatches
import scipy as sypy
from scipy import signal
from scipy import io
from scipy.stats import mode


from ipynb.fs.full.fastKNN import getDistance
from ipynb.fs.full.fastKNN import fastKNN

def normalize(x):
    norm= (x- min(x))/max((x-min(x)))
    return norm


# a simple mappin
fruit=('Banana','Apple')

color=('red', 'orange', 'yellow', 'green', 'blue', 'purple')

training_dataset = np.array([
#   weight, color, # seeds, type
       [303,   2,   1,   0],
       [370,   0,   2,   1],
       [298,   2,   1,   0],
       [277,   2,   1,   0],
       [377,   3,   2,   1],
       [299,   2,   1,   0],
       [382,   0,   2,   1],
       [374,   3,   6,   1],
       [303,   3,   1,   0],
       [309,   2,   1,   0],
       [359,   0,   2,   1],
       [366,   0,   4,   1],
       [311,   2,   1,   0],
       [302,   2,   1,   0],
       [373,   3,   4,   1],
       [305,   2,   1,   0],
       [371,   2,   6,   1]
],dtype=np.float32
)

validation_dataset =np.array([
       [301, color.index('green'),1],
       [346 ,color.index('yellow'), 4],
       [290, color.index('red'), 2 ]
    ],dtype=np.float32
)

    
normalize_datasets=1;
[row,col]=np.shape(training_dataset)

if(normalize_datasets):
#     normalize = @(x) (x - min(x)) / max((x - min(x))); % reduce by smallest value
    for i in range(col-1):
        training_dataset[::,i]=normalize(training_dataset[::,i]);
        validation_dataset[::,i]=normalize(validation_dataset[::,i]);

    


[classified_type, k, index]=fastKNN(training_dataset,validation_dataset);


for i in range(0,len(classified_type)):
    print(fruit[classified_type[i]])

