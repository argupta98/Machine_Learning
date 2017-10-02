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

#define global variables for neural net structure
#hidden layers indexed by layer number, with value as num nodes in layer
layers=[400,200,300]
num_out=10
layers.append(num_out)
num_layers= len(layers)


#placeholders for our X training data and y labels
X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float")

#define the full model for the neural net for an arbitrary number of layers
def nn_model(data):
    #define layer dictionaries to keep together weights and biases, initialized to random
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

