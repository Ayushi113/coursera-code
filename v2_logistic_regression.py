#usr/bin/env/python
import matplotlib.pyplot as plt
import pylab
import numpy as np
from pylab import scatter, show, legend, xlabel, ylabel
from numpy import loadtxt, zeros, ones, array, linspace, logspace
import math

#reading the data and separating into features outputs
regression_data=np.genfromtxt('data_log.txt', dtype=float)
col1=regression_data[:, 0]
col2=regression_data[:, 1]
col3=regression_data[:, 2]

m=len(regression_data)


features=regression_data[:, 0:2]
prediction=regression_data[:, 2]
#this is just an array of ones, to be multiplied as bias
intercept_column=np.ones((m,1), dtype=float)



# ********************* visualizing the data begins ***************** #
#visualization part.
passed = np.where(prediction==1)
failed = np.where(prediction==0)
scatter(features[passed, 0], features[passed, 1], marker='o', c='b')
scatter(features[failed, 0], features[failed, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')

legend(['Not Admitted', 'Admitted'])
show()
# ********************* visualization over ************************#


# ********************* scientific coding begins ***************** #

#initializing the weights in a matrix
# all the weights are zero, to start with.
theta=np.zeros((3, 1), dtype=float)

#taking transpose for multiplication
theta_T=theta.transpose()

#declaring the sig function. This will give the hypothesized 
# value of x.
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


#multiplying the stack
features_up = np.hstack((features, intercept_column))


#calculating error. simply, hypothesized - actual. 
def calc_error(dframe, theta):
    prediction_errors = []

    theta_T=theta.transpose()
    print theta_T.shape
    
# data point = sigmoid(feature vector * weight matrix transpose)
# error computation
    for data_point, index in zip(features_up, range(len(prediction))):
        #print data_point, data_point.shape

        thetaT_x=theta_T.dot(data_point)
        hyp_x=sigmoid(thetaT_x)
        
        error = hyp_x - prediction[index]
      
        prediction_errors.append(error)
    return prediction_errors

#calc_error(regression_data, theta)


#cost function. for each data point, new cost. to be seen decreasing.
def cost_function(dframe, theta):
    CF=[]
    theta_T=theta.transpose()
  
    for data_point, index in zip(features_up, range(len(prediction))):
        
        thetaT_x=theta_T.dot(data_point)
        
        hyp_x=sigmoid(thetaT_x)
        
        cost_function = prediction[index]*np.log(hyp_x) + (1-prediction[index])*np.log(1-hyp_x)
        #print cost_function
        CF.append(cost_function)

    cost_J = - 0.01 * sum(CF)
    return cost_J


#updating theta value using differential. delta theta. 
#differential directly used. 
def theta_update(dframe, theta):
    CFS=[]
#    cost = (1./m) * (-transpose(prediction).dot(log(sigmoid(.dot(theta)))) - transpose(1-prediction).dot(log(1-sigmoid(X.dot(theta)))))
    for epoch, index in zip(range(500), prediction):

        errors = calc_error(dframe, theta)
        
        errors=np.asarray(errors).squeeze()

        #print errors.ndim, errors.shape, col1.ndim, col1.shape

        errors_feat1 = errors*col1 
        #print errors_feat1
        errors_feat2 = errors*col2
     #   errors_feat = errors.dot(dframe)

      #  theta = theta - 0.003 * 1/m * errors_feat

#vectorize both lin and log regression. 
        theta[0][0] = theta[0][0] - 0.01* 1/m * sum(errors)
        theta[1][0] = theta[1][0] - 0.01* 1/m * sum(errors_feat1)
        theta[2][0] = theta[2][0] - 0.01* 1/m * sum(errors_feat2)

        cost = cost_function(dframe, theta)
        print theta
        CFS.append(cost)
    return CFS
    

gradient_des=theta_update(features_up, theta)
print "here"
print gradient_des
#print sorted(co)
plt.plot(gradient_des, "ro")
pylab.xlim([-5,200])
pylab.ylim([-1, 2])
plt.show()





