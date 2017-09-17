# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 18:08:15 2017

@author: Arjun

Handwrittten Neural Net with 3 layers to create a binary classifier. Code is 
adapted from this tutorial:
http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/

"""

import numpy as np

np.random.seed(0)
X, y= sklearn.datasets.make_moons(200, noise=0.20)
