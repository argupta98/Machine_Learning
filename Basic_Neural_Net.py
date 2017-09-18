# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:08:15 2017

@author: Arjun

Handwrittten Neural Net with 3 layers to create a binary classifier. Code is 
adapted from this tutorial:
http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

"""

#Import Packages

#import numpy for fast matrix computations
import numpy as np

#import sklearn for datasets
import sklearn.datasets

#import matplotlib for graphing utility
import matplotlib.pyplot as plt

#import to get a normal linear model for comparison
import sklearn.linear_model


"""
Generate dataset for testing. sklearn has a ton of great datasets that are easy 
to access
"""
np.random.seed(0)
X, y= sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

"""
Train a out-of-the-box linear classifier as a baseline for comparison
"""
# Use sklearn to get baseline linear classifier for comparison
linear_classifier=sklearn.linear_model.LogisticRegressionCV()
linear_classifier.fit(X,y) 

#function for plotting decision function
def plot_decision_boundary(pred_func): 
    # Set min and max values and give it some padding 
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5 
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral) 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral) 
    
# Graph the trained classifier
plot_decision_boundary(lambda x: linear_classifier.predict(x))

"""
Setup functions to Train and Test Neural Net
"""
#NN structural parameters
# 2 dimentional points, so input layer should be 2 dimentional
input_dim=2
# can do 1 dimentionally, but for sake of later augmentation, one node per output
output_dim=2
#controls the rate of decent (can be made better with adaptive gradient decent)
epsilon = 0.01
#Controls how much we are regularizing per iteration
reg_lambda = .01

def train_nn(hidden_dim, data=X, labels=y, iterations=2000):
    
    #initialize with random weights
    np.random.seed(0)
    
    #mapping from input layer to hidden layer
    Theta_1=np.random.randn(input_dim, hidden_dim)
    b_1=np.zeros((1, hidden_dim))
    
    #mapping from hidden layer to output layer
    Theta_2=np.random.randn(hidden_dim, output_dim)
    b_2=np.zeros((1,output_dim))
    
    #run gradient decent to train neural net
    for i_train in range(iterations):
        
        #compute current prediction (forward propagation)
        out_layer_1=data.dot(Theta_1)+b_1
        out_layer_1=np.tanh(out_layer_1)
        out_layer_2=out_layer_1.dot(Theta_2)+b_2
        out_final=np.exp(out_layer_2)
        probs=out_final / np.sum(out_final, axis=1, keepdims=True)
        
        #compute gradients for gradient descent (backpropagation)
        #start at end node, work backwards starting at third layer
        delta_3=probs
        delta_3[range(len(data)), labels]-=1
        dT_2=(out_layer_1.T).dot(delta_3)
        db_2=np.sum(delta_3, axis=0, keepdims=True)
        delta_2=delta_3.dot(Theta_2.T)*(1-np.power(out_layer_1, 2))
        dT_1=np.dot(data.T, delta_2)
        db_1=np.sum(delta_2, axis=0)
        
        #regularize terms
        dT_1+=reg_lambda*Theta_1
        dT_2+=reg_lambda*Theta_2
        
        #update weights from gradient descent
        Theta_1+=-epsilon*dT_1
        Theta_2+=-epsilon*dT_2
        b_1+=-epsilon*db_1
        b_2+=-epsilon*db_2
        
    return {'Theta_1':Theta_1, 'Theta_2': Theta_2, 'b_1':b_1, 'b_2':b_2}

def predict(model, data):
    Theta_1=model['Theta_1']
    Theta_2=model['Theta_2']
    b_2=model['b_2']
    b_1=model['b_1']
    out_layer_1=data.dot(Theta_1)+b_1
    out_layer_1=np.tanh(out_layer_1)
    out_layer_2=out_layer_1.dot(Theta_2)+b_2
    out_final=np.exp(out_layer_2)
    return out_final / np.sum(out_final, axis=1, keepdims=True) 
    
def test(model, data, labels):
    probs=predict(model, data) 
    log_probs=-np.log(probs[range(len(data)),labels])
    error=np.sum(log_probs)
    return 1-(1/len(data))*error

"""
Train and Test Model
"""
model=train_nn(3, iterations=20000)
plot_decision_boundary(lambda x: np.argmax(predict(model, x), axis=1))
print("Accuracy: ",test(model,X,y))