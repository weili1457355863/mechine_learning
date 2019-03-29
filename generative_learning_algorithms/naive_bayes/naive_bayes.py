"""
  author: lw
  email: hnu-lw@foxmail.com
  data: 19-3-28 下午7:26 
  description:
"""
import numpy as np
import pandas as pd
from email_content_filter import email_content_filter
from dictionary import to_dict,create_a_dict,combined
from sklearn.model_selection import train_test_split

#prepare data
email=pd.read_csv("./data/spam-utf8.csv")#为一个datafram
# label={'ham':'0','spam':'1'}
label={'ham':0,'spam':1}
email=email.replace({'v1':label})
X=email_content_filter(email['v2'])
Y=email['v1'].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print(type(Y_train))
#prepare dictionary
dic_1=to_dict('./words/words_alpha.txt')
dic_2=create_a_dict('./data/spam-utf8.csv')
dic=combined(dic_1,dic_2)
print("The dictionary length is:",len(dic))

#convert email to vector
def convert_vector(x):
    x_vector=np.zeros((len(dic),1))
    for i in x:
        try:
            dic[i]
        except Exception:
            pass
        else:
            num_row=dic[i]
            # print(num_row)
            x_vector[num_row]=1
    # print(x_vector)
    # print('x_vector.sum()',x_vector.sum())
    return x_vector

#compute naive bayes model parameters
def compute_model_parameters(X,Y):
    fai_y_0=np.zeros((len(dic),1),dtype=np.float32)
    fai_y_1 = np.zeros((len(dic), 1), dtype=np.float32)
    p_y=0
    x_0=np.zeros((len(dic),1))
    x_1 = np.zeros((len(dic), 1))
    num_y_0 = 0
    num_y_1 = 0
    for x,y in zip(X,Y):
        x_temp=convert_vector(x)
        # print(y)
        if y:
            # x_1+=x_temp
            x_1=np.add(x_1,x_temp)
            num_y_1+=1
        else:
            x_0+=x_temp
            num_y_0+=1
    # print('x_1.sum()',x_1.sum())
    #laplace smoothing
    fai_y_1=(x_1+1)/(num_y_1+2)
    fai_y_0=(x_0+1)/(num_y_0+2)
    p_y=num_y_1/len(Y)
    return fai_y_0,fai_y_1,p_y

#predict whether an email is a spam
def predict(x,fai_y_0,fai_y_1,p_y):
    x_temp=convert_vector(x)
    nums_row=[]
    p_1=1
    p_0=1
    for i in x:
        try:
            dic[i]
        except Exception:
            pass
        else:
            nums_row.append(dic[i])
    for j in nums_row:
        p_1 *=fai_y_1[j]**x_temp[j]
        p_0 *= fai_y_0[j] ** x_temp[j]
    prediction=p_1*p_y/(p_1*p_y+p_0*(1-p_y))
    return prediction
#test data
def test(X,Y,fai_y_0,fai_y_1,p_y):
    correct_amount_spam_email=0
    for x, y in zip(X, Y):
        prediction=predict(x,fai_y_0,fai_y_1,p_y)
        if((prediction>0.9) and y):
            correct_amount_spam_email+=1
            print("this is a spam and the prediction is right")
        if((prediction<0.9) and y):
            print("this is a spam but the prediction is wrong")
        if(y==0):
            print("this is a ham")
    acc=correct_amount_spam_email/Y_test.sum()
    print("The prediction accuracy is:",acc)

fai_y_0,fai_y_1,p_y=compute_model_parameters(X_train,Y_train)
print("The naive bayes model parameters:\n","fai_y_0 is :",fai_y_0,'\n',"fai_y_1 is :",fai_y_1,'\n',"p_y is :",p_y,'\n')
test(X_test,Y_test,fai_y_0,fai_y_1,p_y)




