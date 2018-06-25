# -*- coding: utf-8 -*-
"""
Created on Thu May 31 11:42:36 2018

@author: Lenovo
"""


import pandas as pd
import matplotlib.pyplot as plt


#read data
data=pd.read_csv("Bahubali2_vs_Dangal.csv")
features=data.iloc[:,0:1].values
labels=data.iloc[:,-1].values   #dangal movie
target=data["Bahubali_2_Collections_Per_day"]

#split teh data into train and test
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.2,random_state=0)

#linear regression
from sklearn.linear_model import LinearRegression
reg1=LinearRegression()
reg2=LinearRegression()

reg1.fit(features_train,labels_train)


#predict label
labels_pred1=reg1.predict(10)

#score for days and dangal collection
score1=reg1.score(features_test,labels_test)

#visual understanding training set
plt.scatter(features_train,labels_train,color='red')
plt.plot(features_train,reg1.predict(features_train),color='blue')
plt.title('day vs Dangal(training set)')
plt.xlabel('day')
plt.ylabel('Dangal')
plt.show()


#visual understandin of test set
plt.scatter(features_test,labels_test,color='red')
plt.plot(features_train,reg1.predict(features_train),color='blue')
plt.title('day vs Dangal(test set)')
plt.xlabel('day')
plt.ylabel('Dangal')
plt.show()




#split teh data into train and test
from sklearn.model_selection import train_test_split
features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=0.2,random_state=0)

#linear regression

reg2.fit(features_train,target_train)

#predict results for bahubali2
labels_pred2=reg2.predict(10)

#score for the days and bahubali2 collection
score2=reg2.score(features_test,target_test)

#visual understanding training set
plt.scatter(features_train,target_train,color='red')
plt.plot(features_train,reg2.predict(features_train),color='blue')
plt.title('day vs bahubali1(training set)')
plt.xlabel('day')
plt.ylabel('bahubali2')
plt.show()


#visual understandin of test set
plt.scatter(features_test,target_test,color='red')
plt.plot(features_train,reg2.predict(features_train),color='blue')
plt.title('day vs bahubali2(test set)')
plt.xlabel('day')
plt.ylabel('bahubali2')
plt.show()

