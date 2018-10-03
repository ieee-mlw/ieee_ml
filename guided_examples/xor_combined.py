#!/usr/bin/env python
# coding: utf-8

# # XOR with TensorFlow

# In[1]:


#importing tensorflow
import tensorflow as tf
#import the time module
import time
#importing numpy
import numpy as np
#importing debug library
from tensorflow.python import debug as tf_debug


# In[2]:


#creating a session object which creates an environment where we can execute Operations and evaluate Tensors
sess = tf.Session()


# ## Debugger
#
# ### Uncomment the below line and execute the code to run the debugger.
#
# ### Go to the link once you start execution    			http://localhost:6006/

# In[3]:


#Uncomment the below line to run the debugger
# sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064")


# In[4]:


#Inserting a placeholder for a tensor equal to size of data
X = tf.placeholder(tf.float32, shape=[4,2], name = 'X')

#Inserting a placeholder for a tensor equal to size of labels of the data
Y = tf.placeholder(tf.float32, shape=[4,1], name = 'Y')


# In[5]:


#declaring a variable which will retain its state through multiple runs with random values from normal distribution
W = tf.Variable(tf.truncated_normal([2,2]), name = "W")

#declaring a variable which will retain its state through multiple runs with random values from normal distribution
w = tf.Variable(tf.truncated_normal([2,1]), name = "w")
tf.summary.histogram("hiddenlayer1", W)
tf.summary.histogram("hiddenlayer1", w)
# In[6]:


#declaring a variable which will retain its state through multiple runs with zeros, shape = 4 x 2
c = tf.Variable(tf.zeros([4,2]), name = "c")

#declaring a variable which will retain its state through multiple runs with zeros, shape = 4 x 1
b = tf.Variable(tf.zeros([4,1]), name = "b")


# In[7]:


#define a python operation for the hidden layer
with tf.name_scope("hidden_layer") as scope:
    #the operation of the hidden layer, matrix multpilaction, addition and relu activation function
    h = tf.nn.relu(tf.add(tf.matmul(X, W),c))


# In[8]:


#define a python operation for output layer
with tf.name_scope("output") as scope:
    #the operation at the outplut layer, matrix multiplication, addition and sigmoid activation
    y_estimated = tf.sigmoid(tf.add(tf.matmul(h,w),b))
    tf.summary.histogram("hiddenlayer2", y_estimated)

# In[9]:


#define a python operation for the loss function
with tf.name_scope("loss") as scope:
    #the operation that calculates the loss for our model, here it's the squared loss
    loss = tf.reduce_mean(tf.squared_difference(y_estimated, Y))
    tf.summary.scalar("loss", loss)

# In[10]:


#define a python operation for training our model
with tf.name_scope("train") as scope:
    #the train step with gradient descent optimizer to minimize the loss
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


# In[11]:


#input data
INPUT_XOR = [[0,0],[0,1],[1,0],[1,1]]
#expected output/labels for the data
OUTPUT_XOR = [[0],[1],[1],[0]]

#python operation to initialize the global variables
#init = tf.global_variables_initializer()

#write the summary protocol buffers to event files


#run the graph fragment to execute the operation (initialize global vars)
#sess.run(init)


# In[12]:


#start the clock to record the execution time
t_start = time.clock()
with tf.Session() as sess:

    # Step 10 create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.summary.FileWriter("./logs/xor_logs_final1", sess.graph)
    merged = tf.summary.merge_all()
    tf.initialize_all_variables().run()
#run the model for multiple epochs
    for epoch in range(1000001):

        #run the graph fragment to execute the operation (training)
        #and evaluate each tensor using data from feed_dict
        sess.run(train_step, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})
        #merge = tf.summary.merge_all()
        #check if the step is a multiple of 10000


        #print the char 80 times, forms a separator
        print("_"*80)

        #print the epoch number
        print('Epoch: ', epoch)

        #print y_estimated
        print('   y_estimated: ')

        #run the graph fragment to execute the operation (y_estimated)
        #and evaluate each tensor using data from feed_dict
        for element in sess.run(y_estimated, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR}):
            #print each value of y_estimated
            print('    ',element)


        #print W (theta1)
        print('   W: ')
        #run the graph fragment to execute the operation (W)
        for element in sess.run(W):
            #print each value from W
            print('    ',element)


        #print c(bias1)
        print('   c: ')
        #run the graph fragment to execute the operation (c)
        for element in sess.run(c):
            #print each value from c
            print('    ',element)


        #print w(theta2)
        print('   w: ')
        #run the graph fragment to execute the operation (w)
        for element in sess.run(w):
            #print each value from w
            print('    ',element)


        #print b(bias2)
        print('   b ')
        #run the graph fragment to execute the operation (b)
        for element in sess.run(b):
            #print each value from b
            print('    ',element)

        #merge = tf.summary.merge_all()
        summary, loss_val = sess.run([merged,loss], feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})
        writer.add_summary(summary,epoch)
        #run the graph fragment to execute the operation (loss)
        #and evaluate each tensor using data from feed_dict, print the loss
        #print('   loss: ', sess.run(loss, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR}))

    #end the clock recording the execution time
    t_end = time.clock()


    #print the char 80 times, forms a separator
    print("_"*80)

    #print the execution time
    print('Elapsed time ', t_end - t_start)


    # In[ ]:
