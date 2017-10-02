# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:51:03 2017

@author: Arjun

Basic Neural Net written in Tensorflow to do image classification on MNIST 
data set

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#load data
mnist=input_data.read_data_sets("/temp/data/", one_hot=True)

#placeholders for our X training data and y labels
X = tf.placeholder("float", [None, 784])
y = tf.placeholder("float")

#define the full model for the neural net for an arbitrary number of layers
def nn_model(data, layers):
    #define layer dictionaries to keep together weights and biases, initialized to random
    num_layers= len(layers)
    layer_list=[]
    num_prev_nodes=784
    for i in range(num_layers):
        num_nodes=layers[i]
        #make a dictionary object to hold weights nd biases together
        layer_list.append({"wieghts": tf.variable(tf.random_normal([num_prev_nodes, num_nodes])), "biases": tf.variable(tf.random_normal([num_nodes]))})
        num_prev_nodes=num_nodes
    
    #define layer connections
    current_compute=data
    for i in range(num_layers):
        #output from passing through current layer theta*x+bias
        current_compute = tf.add(tf.matmul(layer_list[i]["weights"],current_compute), layer_list[i]["biases"])
        if(i != len(layer_list)-1):
           #pass through rectified linear function if not last layer
           current_compute = tf.nn.relu(current_compute)
    
    return current_compute
        

#define the training session
def train_nn(data, layers, batch_size=100, iterations=10):
    model=nn_model(data, layers)
    #use cross entropy function for cost 
    cost_func= tf.reduce_mean(tf.nn.softmax.cross_entropy_with_logits(logits=model, labels=y))
    #Use AdamOptimizer to optimize cost func
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)
    with tf.Session as sess:
        sess.run(tf.initialize_all_variables())
        #train model
        for i in range(iterations):
            loss=0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                iter_X, iter_y=mnist.train.next_batch(batch_size)
                sub_loss=sess.run([optimizer, cost_func], feed_dict={X:iter_X, y:iter_y})[1]
                loss+=sub_loss
            print('Iteration', i, 'of', iterations, 'loss: ', loss)
            
        #evaluate model
        is_correct= tf.equal(tf.argmax(model, 1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(is_correct), "float")
        print('Accuracy: ', accuracy.eval({X:mnist.test.images, y:mnist.test.labels}))
        
#define neuarl net architecture
#hidden layers indexed by layer number, with value as num nodes in layer
layers=[400,200,300]
#10 numbers to classify
num_out=10        
#add on the last layer as an output layer
layers.append(num_out)
# run training session
train_nn(X,layers)