"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 19-3-24 下午7:44 
  description:GDA:p(x/y) multivariate normal distribution
"""
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


#creat data
X,Y=make_blobs(n_samples=300,n_features=2,centers=2,random_state=3)
X_train, X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
#change data
X_train=X_train.T
Y_train=Y_train.reshape(Y_train.shape[0],1)
X_test=X_test.T
Y_test=Y_test.reshape(Y_test.shape[0],1)
print(X_train.shape)
print(Y_train.shape)
H_test=np.zeros([Y_test.shape[0],1],dtype=Y_test.dtype)

#compute GDA model parameters
def compute_parameters(X,Y):
    num_y_0=np.sum(Y==0)#统计数组中出现某个值的个数
    num_y_1=np.sum(Y==1)
    fai=num_y_1/Y.shape[0]
    print("GDA model parameters:")
    print("fai=",fai)
    miu_0=np.dot(X,(1-Y))/num_y_0
    print("miu_0=",miu_0)
    miu_1=np.dot(X,(Y))/num_y_1
    print("miu_1=",miu_1)
    miu_y=np.dot(miu_0,(1-Y).T)+np.dot(miu_1,Y.T)
    print(miu_y.shape)
    cov=np.dot((X-miu_y),(X-miu_y).T)/Y.shape[0]
    print("cov=",cov)
    return fai,miu_0,miu_1,cov

#predict new data
def predict(x,miu_0,miu_1,cov):
    p_0=np.exp(-0.5*(np.dot(np.dot((x-miu_0).T,np.linalg.inv(cov)),(x-miu_0))))/(2*np.pi*np.sqrt(np.linalg.det(cov)))
    p_1 = np.exp(-0.5 * (np.dot(np.dot((x - miu_1).T, np.linalg.inv(cov)), (x - miu_1)))) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    if(p_0>p_1):
        y_=0
    else:
        y_=1
    return y_
#

#test model ability
def testParameter(X,Y,miu_0,miu_1,cov):
    i=0
    for x, y in zip(X.T, Y):  # every step all samples
        x=x.reshape(2,1)
        # print("x.shape",x.shape)
        # print("y.shape", y.shape)
        # print("cov.shape",cov.shape)
        H_test[i] = predict(x,miu_0,miu_1,cov)
        i=i+1
    plt.cla()
    plt.scatter(X[0, :], X[1, :], c=H_test[:,0], edgecolors='white', marker='s')
    plt.show()
#accuracy
def computeAccuracy(A,Y):
    temp=A ^ Y
    print(A)
    print(Y)
    print(temp.sum())
    accracy=(A.shape[1]-temp.sum())/A.shape[1]*100
    return accracy

fai,miu_0,miu_1,cov=compute_parameters(X_train,Y_train)
testParameter(X_test,Y_test,miu_0,miu_1,cov)
acc=computeAccuracy(H_test,Y_test)
print("accuracy:",acc)