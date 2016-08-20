#usr/bin/env/python
import matplotlib.pyplot as plt
import pylab
import numpy as np
from pylab import scatter, show, legend, xlabel, ylabel
from numpy import loadtxt, zeros, ones, array, linspace, logspace
import math

regression_data=np.genfromtxt('data_log.txt', dtype=float)
col1=regression_data[:, 0]
col2=regression_data[:, 1]

m=len(regression_data)
features=regression_data[:, 0:2]
prediction=regression_data[:, 2]
intercept_column=np.ones((m,1), dtype=float)



passed = np.where(prediction==1)
failed = np.where(prediction==0)
scatter(features[passed, 0], features[passed, 1], marker='o', c='b')
scatter(features[failed, 0], features[failed, 1], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
#show()



def sigmoid(x):
  return 1 / (1 + math.exp(-x))

theta=np.zeros((3, 1), dtype=float)
theta_T=theta.transpose()

features_up = np.hstack((features, intercept_column))

print theta, theta_T

def calc_error(dframe, theta_T):
    prediction_errors = []
    theta_T=theta.transpose()
  
    for data_point, index in zip(features_up, range(len(prediction))):
        thetaT_x=theta_T.dot(data_point)
        hyp_x=sigmoid(thetaT_x)
      
        error = prediction[index]*math.log(hyp_x) + (1-prediction[index])*math.log(1-hyp_x)
      
        prediction_errors.append(error)
    return prediction_errors

calc_error(regression_data, theta_T)



def theta_update(dframe, theta_T):
#    cost = (1./m) * (-transpose(prediction).dot(log(sigmoid(.dot(theta)))) - transpose(1-prediction).dot(log(1-sigmoid(X.dot(theta)))))
    for data_point in dframe:
        errors = calc_error(dframe, theta)
        errors_intercept=numpy.asarray(errors)
        errors_feat1 = errors_intercept*col1
        errors_feat2 = errors_intercept*col2
    

        theta[0][0] = theta[0][0] - 0.003* 1/m * sum(errors_intercept)
        theta[0][1] = theta[0][0] - 0.003* 1/m * sum(errors_feat1)
        theta[0][2] = theta[0][0] - 0.003* 1/m * sum(errors_feat2)

        cost = 1/m * sum(errors_intercept)
    return 1







