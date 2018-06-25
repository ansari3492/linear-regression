# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:18:54 2018

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt


#read data
data=pd.read_csv("Foodtruck.csv")
features=data.iloc[:,0:1].values
labels=data.iloc[:,-1].values

#split teh data into train and test
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.1,random_state=0)

#linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(features_train,labels_train)

#predict results for jaipur 3.073 million
labels_pred=reg.predict(3.073)

#score
score=reg.score(features_test,labels_test)

#visual understanding training set
plt.scatter(features_train,labels_train,color='red')
plt.plot(features_train,reg.predict(features_train),color='blue')
plt.title('population vs profit(training set)')
plt.xlabel('population')
plt.ylabel('profit')
plt.show()


#visual understandin of test set
plt.scatter(features_test,labels_test,color='red')
plt.plot(features_train,reg.predict(features_train),color='blue')
plt.title('population vs profit(test set)')
plt.xlabel('population')
plt.ylabel('profit')
plt.show()




