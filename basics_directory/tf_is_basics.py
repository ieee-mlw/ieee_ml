import numpy as np
import tensorflow as tf

#A TensorFlow Session for use in interactive contexts, such as a shell.
sess = tf.InteractiveSession()

#a place in memory where we will store value later on
x = tf.placeholder("float", [1, 3])
# deep layers
num_layers = 2
for layer in range(num_layers):
  with tf.name_scope('relu'):#Help specify hierarchical names
    w=tf.Variable(tf.random_normal([3,3]),name='w')#define a variable
    b = tf.Variable(tf.zeros([1, 3]))#bias
    y = tf.matmul(x, w)+b
    relu_out = tf.nn.relu(y)#activate function
    
with tf.name_scope('softmax'):  
  softmax_w = tf.Variable(tf.random_normal([3, 3]))
  softmax_b = tf.Variable(tf.zeros([1, 3]))
  logit = tf.matmul(relu_out, softmax_w)+ softmax_b
  softmax = tf.nn.softmax(logit)#Make predictions for n targets that sum to 1

# Returns an Op that initializes global variables.
sess.run(tf.global_variables_initializer())
#Visualize the graph
writer = tf.summary.FileWriter(
'/tmp/tf_logs', sess.graph) 

#Labels
answer = np.array([[0.0, 1.0, 0.0]])

#variable to store Labels
labels = tf.placeholder("float", [1, 3])

#Loss function for softmax
l2reg = tf.reduce_sum(tf.square(softmax_w))#L2 regularize
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
logits=logit, labels=labels, name='xentropy')
loss=l2reg+cross_entropy

#GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_op = optimizer.minimize(loss)

#Feed
for step in range(10):
    _, result=sess.run([train_op,softmax],feed_dict={x:np.array([[1.0, 2.0, 3.0]]), labels:answer}) 
    print(result)
#close sesstion
sess.close()

