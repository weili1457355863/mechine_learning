"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 19-3-17 下午10:05 
  description:softmax regression k=3
"""
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
from math import exp,log

#creat data
X,Y=make_blobs(n_samples=300,n_features=2,centers=3,random_state=3)
# print(Y)
# transform y into one-hot vector
# encoder=OneHotEncoder(categorical_features='auto')
encoder=OneHotEncoder()
Y=encoder.fit_transform(np.reshape(Y,(Y.shape[0],1))).toarray()
# print(Y)
X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
print(X_train.shape)
# show data
plt.scatter(X[:,0],X[:,1],c=Y,edgecolors='white')
plt.show()
# change data shape, add column 1 at head
X_train_1=np.ones([X_train.shape[0],1],dtype=X_train.dtype)
X_train=np.concatenate((X_train_1,X_train),axis=1)
X_test_1=np.ones([X_test.shape[0],1],dtype=X_train.dtype)
X_test=np.concatenate((X_test_1,X_test),axis=1)

#parameters
theta=np.random.normal(0,0.1,9).reshape(3,3)
lr=0.05
decay_rate=0.005
LOSS = []
H_train=np.zeros([Y_train.shape[0],3],dtype=Y_train.dtype)
H_test=np.zeros([Y_test.shape[0],3],dtype=Y_test.dtype)
#compute Hypothesis
def softmaxRegression(theta,x):
    temp=np.dot(theta.T,x)
    temp=np.exp(temp)
    sum=np.sum(temp)
    h=temp/sum
    return h
#Update parameters
def updateParameters(x,y,y_train,theta,lr):
    y_train=np.reshape(y_train, (3, 1))
    y=np.reshape(y, (3, 1))
    label=np.argmax(y,axis=0)#一列中所有行最大的数
    theta[:, label] = theta[:, label] +lr * (y[label][0] - y_train[label][0] )* x #效果更好一点
    # theta[:, label] = theta[:, label] - lr * (-y[label][0] * (1 / y_train[label][0] * x))
    return theta
#Compute loss
def computeLoss(X,Y,H):
    loss=0
    for x, y,h in zip(X, Y,H):
        y=np.reshape(y,(3,1))
        h= np.reshape(h, (3, 1))
        label = np.argmax(y, axis=0)
        loss+=-y[label][0]*log(h[label][0]+0.000001)#avoid h=0
    loss/=X.shape[0]
    return loss
#SGD
def stochasticGradientDescent(theta):
    global lr
    for epoch in range(20):
        i=0
        # lr=(1/(1+decay_rate*epoch))*lr
        for x, y in zip(X_train, Y_train):  # every step all samples
            x = np.reshape(x, (3,1))
            H_train[i]= softmaxRegression(theta, x).T#threshold function
            theta=updateParameters(x,y,H_train[i],theta,lr)
            loss=computeLoss(X_train,Y_train,H_train)
            LOSS.append(loss)
            print(epoch,"-",i,"loss=",loss)
            i=i+1
        plt.cla()
        plt.scatter(X_train[:,1],X_train[:,2], c=np.argmax(H_train,axis=1), edgecolors='white',marker='s')
        # plt.plot(p_x, theta[1, 0] * p_x + theta[0, 0], c='orange')
        plt.pause(0.01)
    return theta
def test(X,Y,theta):
    i=0
    for x, y in zip(X, Y):  # every step all samples
        x = np.reshape(x, (3, 1))
        H_test[i] = softmaxRegression(theta, x).T  # threshold function
        i=i+1
    plt.cla()
    plt.scatter(X[:, 1], X[:, 2], c=np.argmax(H_test, axis=1), edgecolors='white', marker='s')
    plt.show()
theta=stochasticGradientDescent(theta)
test(X_test,Y_test,theta)
# plt.plot(LOSS,c='r')
# plt.show()


