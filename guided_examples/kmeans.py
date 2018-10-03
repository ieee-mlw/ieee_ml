
# coding: utf-8

# # K-Means Example
#
# Implement K-Means algorithm with TensorFlow, and apply it to classify
# handwritten digit images. This example is using the MNIST database of
# handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).
#
# Note: This example requires TensorFlow v1.1.0 or over.
#
# - Author: Aymeric Damien
# - Project: https://github.com/aymericdamien/TensorFlow-Examples/

# In[1]:


#importing print function from python3 to python2
from __future__ import print_function

#importing numpy
import numpy as np
#importing tensorflow
import tensorflow as tf
#the graph for k-means clustering
from tensorflow.contrib.factorization import KMeans

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

#ignore all the warnings and don't show them in the notebook
import warnings
warnings.filterwarnings('ignore')
#importing debug library
from tensorflow.python import debug as tf_debug


# In[2]:


# Start TensorFlow session
#creating a session object which creates an environment where we can execute Operations and evaluate Tensors
sess = tf.Session()


# ## Debugger
#
# ### Uncomment the below line and execute the code to run the debugger.
#
# ### Go to the link once you start execution    			http://localhost:6006/

# In[3]:


#Uncomment the below line to run the debugger
#sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:6064",send_traceback_and_source_code=False)

sess = tf_debug.LocalCLIDebugWrapperSession(sess)


# In[4]:


# Import MNIST data
#from tensorflow examples import the mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
#read the input data, perform one hot encoding on it
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#load the entire mnist data
full_data_x = mnist.train.images


# In[5]:


# Parameters
num_steps = 50 # Total steps to train
batch_size = 1024 # The number of samples per batch
k = 25 # The number of clusters
num_classes = 10 # The 10 digits
num_features = 784 # Each image is 28x28 pixels

# Input images
#Inserting a placeholder for a tensor equal to size of data
X = tf.placeholder(tf.float32, shape=[None, num_features])
# Labels (for assigning a label to a centroid and testing)
#Inserting a placeholder for a tensor equal to size of data for labels
Y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means Parameters
#Initialize k means with k clusters and use the cosine distance metric
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine',
                use_mini_batch=True)


# In[6]:


# Build KMeans graph
(all_scores, cluster_idx, scores, cluster_centers_initialized,init_op,train_op) = kmeans.training_graph()
cluster_idx = cluster_idx[0] # fix for cluster_idx being a tuple
avg_distance = tf.reduce_mean(scores)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()


# In[ ]:


# Run the initializer
#run the graph fragment to execute the operation (initialize variables) and evaluate each tensor using data from feed_dict
sess.run(init_vars, feed_dict={X: full_data_x})
#run the graph fragment to execute the operation (initialize clusters) and evaluate each tensor using data from feed_dict
sess.run(init_op, feed_dict={X: full_data_x})

# Training
#Train the algorithm for the steps pre-decided above
for i in range(1, num_steps + 1):
    #run the graph fragment to execute the operation (training, calculating avg distance)
    #and evaluate each tensor using data from feed_dict
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    #check if the step is the first one or a multiple of 10
    if i % 10 == 0 or i == 1:
        #Print the step number if the above condition is true
        print("Step %i, Avg Distance: %f" % (i, d))


# In[ ]:


# Assign a label to each centroid
# Count total number of labels per centroid, using the label of each training
# sample to their closest centroid (given by 'idx')
#creating a numpy array of zeros with shape (k,num_classes)
counts = np.zeros(shape=(k, num_classes))
#run the loop idx length times
for i in range(len(idx)):
    #increment the count matrix with train label values
    counts[idx[i]] += mnist.train.labels[i]
# Assign the most frequent label to the centroid
#find the index of the max value in each row of count matrix
labels_map = [np.argmax(c) for c in counts]
#convert the list created above to a tensor
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
#lookup id of labels using the tensor
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
#compare the predicted label to the true label
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(Y, 1), tf.int32))
#calculate the accuracy of the predictions
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
#load the test data and its corresponding labels
test_x, test_y = mnist.test.images, mnist.test.labels
#run the graph fragment to execute the operation (prediction and accutacy calculation)
#and evaluate each tensor using data from feed_dict, print the accuracy
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
