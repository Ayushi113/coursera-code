#usr/bin/env/python
import matplotlib.pyplot as plt
import pylab, numpy
from numpy import loadtxt, zeros, ones, array, linspace, logspace
import sympy, sys
from sympy import *
regression_data=numpy.genfromtxt('data.txt', dtype=float)

col1=regression_data[:, 0]
col2=regression_data[:, 1]

pylab.xlim([0,20])
pylab.ylim([0,20])

#global declaration of theta_0 and theta_1 values. 
theta=numpy.zeros(shape=(1,2))

def calc_error(dframe, theta):
    list_errors=[]
    predicted_vals=[]

    for data_point in dframe:
    #predictig the hypothesized value at every data point
        hyp_x=theta[0][0]*data_point[0] + theta[0][1]
        #print hyp_x, "hyp val"
     
        #creating a list of hyp values
        predicted_vals.append(hyp_x)
        #calculating error
        error=hyp_x-data_point[1]
        #print error, "err"
        #squaring error
        sq_error=error*error
        #print sq_error, "sqerr"
        #appending to a list
        list_errors.append(error)
    return list_errors

def cost_function(dframe, sum_er):
    m = len(dframe)	  
    #print m
    #print sum_er
    #creating a cost function which takes this list of errors
    J = 0.005154639 * sum_er * sum_er
    #print J, "je"
    return J

init_cost=cost_function(regression_data, theta)

def pred_value(dframe):
    Js=[]
    m = len(dframe)      
   #computing cost function and updating for each data point
    for j in range(len(dframe)):
        errors = calc_error(dframe, theta)
        errors_np=numpy.asarray(errors)
        hello = errors_np * col1
#        print hello
#        exit(0)
        sum_errors=sum(errors)
        #print sum_errors, "sum of sqerrors"
        
        #updating th0 and th1
        theta[0][0] = theta[0][0] - 0.003* 1/m * sum_errors
        theta[0][1] = theta[0][1] - 0.003* 1/m * sum(hello)
        #print theta    
        
        errors = calc_error(dframe, theta)
        sum_errors=sum(errors)
        #print sum_errors
        J=cost_function(dframe, sum_errors)
        J_tuple=[J, theta[0][0], theta[0][1]]
        Js.append(J_tuple)
    
    return Js     


gradient_des=pred_value(regression_data)
print gradient_des[0:4]
sort_gd=sorted(gradient_des, key = lambda x: x[0])
outputs=[]
for item in range(len(sort_gd)):
    slope = sort_gd[0][1]
    intercept = sort_gd[0][2]
    for j in col1:
        y_hat=slope*j + intercept
        outputs.append(y_hat)

plt.plot(outputs, "ro")
pylab.xlim([-5,50])
pylab.ylim([-5, 50])
plt.show()

plt.plot(gradient_des, "ro")
pylab.xlim([-5,80])
pylab.ylim([0,2000])
plt.ylabel('some numbers')
#plt.show()

#cal_slope(regression_data)
plot2=plt.plot(col2, col1, "ro")
#plt.show()


