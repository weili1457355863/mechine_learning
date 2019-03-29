"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 19-3-22 下午2:11 
  description: vectorization logistic regression, high speed
"""
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


#creat data
X,Y=make_blobs(n_samples=300,n_features=2,centers=2,random_state=3)
X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
# show data
# plt.scatter(X[:,0],X[:,1],c=Y,edgecolors='white')
# plt.scatter(X_test[:,0],X_test[:,1],c=Y_test,edgecolors='white')
# plt.show()
# print(X_train.shape)
# print(Y_train.shape)
# change data
X_train=X_train.reshape(X_train.shape[1],X_train.shape[0])
Y_train=Y_train.reshape(1,Y_train.shape[0])
# X_test=X_test.reshape(X_test.shape[1],X_test.shape[0])
X_test=X_test.T
Y_test=Y_test.reshape(1,Y_test.shape[0])
plt.scatter(X_test[0,:],X_test[1,:],c=Y_test[0,:],edgecolors='white')
plt.show()
#parameters
W=np.random.normal(0,0.1,2).reshape(2,1)
b=0
lr=0.01 #learning rate
epoch=100 # iiretitons
def costFunction(A,Y):
    # print(A.shape)
    # print(np.log(A))
    J=np.sum(-Y*np.log(A)-(1-Y)*np.log(1-A))/Y.shape[1]
    return J
# forward propogation
def forwardPropogation(W,b,X):
    Z=np.dot(W.T,X)+b
    A=1/(1+np.exp(Z))
    return A
# backward propogation
def backwardPropogation(W,b,A,X,Y):
    dZ=A-Y
    dW=np.dot(X,dZ.T)/X.shape[1]
    # print("dW.shape:",dW.shape)
    db=np.sum(dZ)/X.shape[1]
    W-=lr*dW
    b-=lr*db
    return W,b
#training
def training(W,b):
    for i in range(epoch):
        A=forwardPropogation(W,b,X_train)
        print(A.shape)
        W,b=backwardPropogation(W,b,A,X_train,Y_train)
        J=costFunction(A,Y_train)
        # print("cost loss=",J)
    return W,b
#accuracy
def computeAccuracy(A,Y):
    temp=A ^ Y
    print(A)
    print(Y)
    print(temp.sum())
    accracy=(A.shape[1]-temp.sum())/A.shape[1]
    return accracy

#testing
W,b=training(W,b)
A=forwardPropogation(W,b,X_test)
A=np.rint(A).astype(np.int64)
accracy=computeAccuracy(A,Y_test)
print("Accuracy:",accracy)
# print(Y_test.shape)
plt.cla()
plt.scatter(X_test[0, :], X_test[1,:], c=A[0,:], edgecolors='white', marker='s')
# plt.scatter(X_test[0, :], X_test[1,:], c=Y_test[0,:], edgecolors='white', marker='s')
plt.show()

