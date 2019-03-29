"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 19-3-17 下午2:57 
  description: logistic function or sigmod function, classification problem{0,1},perceptron learning algorithm,change
  two lines(69,87) if you want to change between sigmod and perceptron learning method
"""
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from math import exp,log

#creat data
X,Y=make_blobs(n_samples=300,n_features=2,centers=2,random_state=3)
X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
# show data
plt.scatter(X[:,0],X[:,1],c=Y,edgecolors='white')
plt.show()
# change data shape, add column 1 at head
X_train_1=np.ones([X_train.shape[0],1],dtype=X_train.dtype)
X_train=np.concatenate((X_train_1,X_train),axis=1)
X_test_1=np.ones([X_test.shape[0],1],dtype=X_train.dtype)
X_test=np.concatenate((X_test_1,X_test),axis=1)

#parameters
theta=np.random.normal(0,0.1,3).reshape(3,1)
lr=0.01
H_train=np.zeros([Y_train.shape[0],1],dtype=Y_train.dtype)
H_test=np.zeros([Y_test.shape[0],1],dtype=Y_test.dtype)
#logistic function or sigmod function
def sigmodFunction(theta,x):
    z=np.dot(theta.T,x)
    h=1/(1+exp(-z))
    # print(h)
    return h
#logistic function or sigmod function
def thresholdFunction(theta,x):
    z=np.dot(theta.T,x)
    if z<0:
        h=0
    else:
        h=1
    return h
# update theta
def updateTheta(theta,gradient,lr):
    theta=theta+lr*gradient
    return theta
#compute log likelihood

def computeLikelihood(X,Y,theta):
    logLike=0
    for x, y in zip(X, Y):  # every step all samples
        x = np.reshape(x, theta.shape)
        h= sigmodFunction(theta, x)
        # print(h)
        if h==1.0:
            h=1-0.00001
        # print(h)
        logLike+=y*log(h)+(1-y)*log(1-h)
    logLike=logLike/(X.shape[0])
    return logLike

#SGD
def stochasticGradientDescent(theta):
    for epoch in range(20):
        i=0
        for x, y in zip(X_train, Y_train):  # every step all samples
            x = np.reshape(x, theta.shape)
            # H_train[i] = sigmodFunction(theta, x)#sigmod function
            H_train[i] = thresholdFunction(theta, x)#threshold function
            gradient = (y-H_train[i]) * x
            theta = updateTheta(theta, gradient,lr)
            likeliHood=computeLikelihood(X_train,Y_train,theta)
            print("epoch=",epoch,",",i,":likelyhood=",likeliHood)
            i=i+1
            # print("theta:=", theta)
        plt.cla()
        plt.scatter(X_train[:,1],X_train[:,2], c=H_train[:,0], edgecolors='white',marker='s')
        # plt.plot(p_x, theta[1, 0] * p_x + theta[0, 0], c='orange')
        plt.pause(0.01)
    return theta
#test model ability
def testParameter(X,Y,theta):
    i=0
    for x, y in zip(X, Y):  # every step all samples
        x = np.reshape(x, theta.shape)
        # H_test[i]=sigmodFunction(theta, x)#sigmod function
        H_test[i] = thresholdFunction(theta, x)#threshold function
        i=i+1
    plt.cla()
    plt.scatter(X[:, 1], X[:, 2], c=H_test[:,0], edgecolors='white', marker='s')
    print(H_test)
theta=stochasticGradientDescent(theta)
testParameter(X_test,Y_test,theta)
plt.show()