#usr/bin/env/python
import matplotlib.pyplot as plt
import pylab, numpy
from numpy import loadtxt, zeros, ones, array, linspace, logspace
import sympy, sys
from sympy import *
regression_data=numpy.genfromtxt('data.txt', dtype=float)

col1=regression_data[:, 0]
col2=regression_data[:, 1]

pylab.xlim([10,20])
pylab.ylim([10,20])

#global declaration of theta_0 and theta_1 values. 
theta=numpy.zeros(shape=(1,2))
print theta[0][0], theta[0][1]

def cost_function(dframe, theta):
    list_errors=[]
    predicted_vals=[]
    m = len(dframe)	  
    
    for data_point in dframe:
    #predictig the hypothesized value at every data point
	hyp_x=theta[0][0]*data_point[0] + theta[0][1]
        print hyp_x, "hyp val"
     
        #creating a list of hyp values
	predicted_vals.append(hyp_x)
        #calculating error
        error=data_point[1]-hyp_x
        print error, "err"
      	#squaring error
      	sq_error=error*error
        print sq_error, "sqerr"
        #appending to a list
        list_errors.append(sq_error)
    #creating a cost function which takes this list of errors
    J = (1.0 / (2 * m)) * sum(list_errors)
    return J, sum(list_errors)


init_cost=cost_function(regression_data, theta)

def pred_value(dframe):
    m = len(dframe)      
   #computing cost function and updating for each data point
    for j in range(5):
        J_init, sum_sqerrors = cost_function(dframe, theta)
        print sum_sqerrors, "sum of sqerrors"
        exit(0)   
        #updating th0 and th1
        theta[0][0] = theta[0][0] - 0.01* 1/m * sum_sqerrors
        theta[0][1] = theta[0][1] - 0.01* 1/m * sum_sqerrors * dframe[j][0]
        print theta    

        J=cost_function(dframe, theta)
        print J[0]
     


pred_value(regression_data)

#cal_slope(regression_data)
plot2=plt.plot(col2, col1, "ro")
plt.show()


