# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:34:05 2019

@author: CNsasi
"""
#importing libraries
import pandas as pd
from sklearn import datasets

#Let's load our data
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target

#Split our dependant and independant feature
X=features.iloc[:,:]
y=targets[:,]

#splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y, test_size=0.30, random_state=0)

#Normalization of variables
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

#Fitting Random Forest Regressor to the Training Test
from sklearn.ensemble import RandomForestRegressor
Regressor=RandomForestRegressor(random_state=0, n_estimators=100)
Regressor.fit(X_train,y_train)

#Predicting the test set results
y_pred=Regressor.predict(X_test)

#measure accuracy
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
