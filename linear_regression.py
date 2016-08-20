#usr/bin/env/python
import matplotlib.pyplot as plt
import pylab
import numpy
import sympy
from sympy import *
regression_data=numpy.genfromtxt('data.txt', dtype=float)
print regression_data
col1=regression_data[:, 0]
col2=regression_data[:, 1]

pylab.xlim([10,20])
pylab.ylim([10,20])
#data_frame = [profits, population]
def cal_slope(dframe):
	for row in range(len(dframe)-1):
		slope = float((dframe[row+1][1]-dframe[row][1])/(dframe[row+1][0]-dframe[row][0]))
		y_intercept=dframe[row+1][1]-slope*dframe[row+1][0]
		print "slope", slope, "y_intercept", y_intercept


def sum_problems(dframe):
   	list_errors=[]
	predicted_vals=[]
	init_slope=3
	init_intercept=2
   	for data_point in dframe:
		predicted=init_slope*data_point[0] + init_intercept
		predicted_vals.append(predicted)
		error=data_point[1]-predicted
      	sq_error=error*error
		list_errors.append(sq_error)
	return predicted_vals, sum(list_errors)

def pred_value(dframe):
    pvals, lse_sum = sum_problems(dframe)
    cost_function=(1/(2*len(dframe)))*lse_sum
    theta_0, theta_1=symbols('theta_0, theta_1')    
    for j in range(len(dframe)):
    	xdiff_cost=diff(cost_function, theta_0)
    	ydiff_cost=diff(cost_function, theta_1)
        slope_update = dframe[j][0] - 0.1*xdiff_cost
        intercept_update = dframe[j][1] - 0.1*ydiff_cost
        print slope_update, intercept_update
		


pred_value(regression_data)

cal_slope(regression_data)
plot2=plt.plot(col2, col1, "ro")
plt.show()


