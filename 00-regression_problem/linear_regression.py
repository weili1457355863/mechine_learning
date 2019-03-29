"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 19-3-17 下午2:57
  description: linear regression learning, include BD, SGD, Normal Equation, Locally weighted linear regression
"""
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from math import exp

#create data
X,y=datasets.make_regression(n_samples=250,n_features=1,noise=20,random_state=0,bias=50)
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
# print(X_train.shape)
#show data
# fig=plt.figure()
# plt.scatter(X_train,y_train,c='red',edgecolors='white')
# plt.scatter(X_test,y_test,c='blue',edgecolors='white')
# # plt.savefig("1.png")
# plt.show()
# change data shape, add column 1 at head
# X_train_1=np.ones([X_train.shape[0],1],dtype=X_train.dtype)
X_train_1=np.ones(X_train.shape,dtype=X_train.dtype)
X_train=np.concatenate((X_train_1,X_train),axis=1)
# print(X_train)
#parameters
theta=np.zeros([2,1],dtype=np.float32)
lr=0.001
#for plotting regression line
p_x=np.linspace(-2.5,2.5,500)
#hypothesis(h(x))
def hypothesis(theta,x):
    h_x=np.dot(theta.T,x)#matrix dot
    # print(h_x)
    return h_x
#Batch Gradient Descent
def updateTheta(theta,gradient,lr):
    theta=theta-lr*gradient
    return theta
#locally weighted
x_point=-1.0
t=0.1
def updateThetaWeithted(theta,gradient,lr,x):
    # print(x[1,0])
    w=exp(-(x[1,0]-x_point)**2/(2*t**2))
    print("x:=", x[1,0])
    print("W:=",w)
    theta=theta-w*lr*gradient
    # theta = theta - lr * gradient
    return theta
x=np.zeros([2,1],dtype=np.float32)
# plt.ion()
def batchGradientDescent(theta):
    gradient = np.zeros([2, 1], dtype=np.float32)
    loss = 0
    for epoch in range(50):
        for x,y in zip(X_train,y_train):#every step all samples
            # x[0,0]=1
            # x[1,0]=x_
            # print(x)
            # print(theta)
            # print(x.shape)
            x = np.reshape(x, theta.shape)
            h_x=hypothesis(theta,x)
            # print(h_x)
            # print(h_x-y_)
            # print(x.shape)
            gradient+=(h_x-y)*x
            #print(gradient)
            loss+=(h_x-y)*(h_x-y)
        loss=loss/(2*X_train.shape[0])
        print(loss)
        theta=updateTheta(theta,gradient,lr)
        loss=0
        plt.cla()
        plt.scatter(X_train[:,1], y_train, c='red', edgecolors='white')
        plt.plot(p_x,theta[1,0]*p_x+theta[0,0],c='orange')
        plt.pause(0.01)
    return theta
#SGD
def stochasticGradientDescent(theta):
    for epoch in range(50):
        for x, y in zip(X_train, y_train):  # every step all samples
            x = np.reshape(x, theta.shape)
            h_x = hypothesis(theta, x)
            gradient = (h_x - y) * x
            # theta = updateTheta(theta, gradient, lr)
            theta = updateThetaWeithted(theta, gradient, lr,x)
            print("theta:=", theta)
        plt.cla()
        plt.scatter(X_train[:,1], y_train, c='red', edgecolors='white')
        plt.plot(p_x, theta[1, 0] * p_x + theta[0, 0], c='orange')
        plt.pause(0.01)
    return theta
#Normal Equations
def normalEquation():
    temp=np.linalg.inv(np.dot(X_train.T,X_train))
    theta=np.dot(np.dot(temp,X_train.T),y_train)
    theta=np.reshape(theta,[2,1])
    print(theta.shape)
    plt.cla()
    plt.scatter(X_train[:, 1], y_train, c='red', edgecolors='white')
    plt.plot(p_x, theta[1, 0] * p_x + theta[0, 0], c='orange')
    plt.pause(0.01)
    return theta
# batchGradientDescent(theta)
stochasticGradientDescent(theta)
# normalEquation()
# plt.ioff()#shutdown 交互模式
plt.show()