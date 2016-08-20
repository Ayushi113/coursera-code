import matplotlib.pyplot as plt
import pylab
import numpy as np
from pylab import scatter, show, legend, xlabel, ylabel
from numpy import loadtxt, zeros, ones, array, linspace, logspace
import math

regression_data=np.genfromtxt('data_log.txt', dtype=float)
m=len(regression_data)

features=regression_data[:, 0:2]
prediction=regression_data[:, 2]
bias_vector=np.ones((m, 1), dtype=float)
ones=np.ones((m, 1), dtype=float)
features_up = np.hstack((features, bias_vector))
#print features_up.shape

#first layer of weights matrix
#question: is this a self-designed number of node assignment in hidden layer1?
#the columns = number of nodes in prev layer(no bias) + 1
theta1=np.zeros((4, 3), dtype=float)
theta1_T=theta1.transpose()

# second layer of weights matrix
# here I can't make it that way. Hacked columns to 4. 
theta2=np.zeros((1, 4), dtype=float)
theta2_T=theta2.transpose()

# activation function
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# 
def calc_error(dframe, theta):
    thetaT_x=dframe.dot(theta1_T)
    #returns mX4 matrix  
    print thetaT_x.shape
    #this is wrong. Can't do this.
    #to_activations=[sum(i) for i in zip(*thetaT_x)]

    #to_activations=np.asarray(to_activations)
    print thetaT_x.shape
    
    activations_1=sigmoid(thetaT_x)
    
    activations_2=activations_1.dot(theta2_T)
    print activations_2
   
    to_activations2=np.asarray(activations_2)
    activs_2=sigmoid(to_activations2)
    print activs_2
    
    
    cost_function = prediction.dot(np.log()) + (ones-prediction).dot(np.log(ones-hyp_x))
    print cost_function


calc_error(features_up, theta1)

    







