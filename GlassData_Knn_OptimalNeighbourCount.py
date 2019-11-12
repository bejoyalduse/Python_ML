# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:03:53 2019

@author: bejoyalduse
"""

# About this study
# Create a model to classify the glass into category types based on the attributes of each glass type
# we use kNN classification from sklearn in python
# kNN classification on the glass.csv data available from kagle
# https://www.kaggle.com/uciml/glass#glass.csv  


import matplotlib
import matplotlib.pyplot as plt # to plot the final scores and determine the optimal parameters
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# read the csv data into variable glass pandas data frame
glass= pd.read_csv("glass_table.csv")
glass.head(6)

# save the attributes into X and predictor (or independent variable)
X = glass.drop("Type", axis=1)

# save the glass type into response variable (or dependent variable)
Y = glass["Type"]

# Lets split the data into training and test data
x_train, x_test,y_train,y_test = train_test_split(X, Y, test_size = .2, random_state=25)#20% hold out for testing

# a glimpse of the training set can be seen by plotting x_train and y_train
# plt.plot(x_train,y_train) 

###
# carry out a parameter tuning based on miss classification error
# carry out a 10-fold cross validation

from sklearn.model_selection import cross_val_score

# set a numeric index 1 to 50 
parametercount = range(1,25)

# predefine an empty list to save cv_Scores
crossval_scores = []

# carry out cross validation and save the accuracy score 
# for 10 fold cross validation for each parameter count

for k in parametercount:
	knn = KNeighborsClassifier(n_neighbors= k)
	scores =cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
	crossval_scores.append(scores.mean())
    

# Determine the misclassification error by subtractin the cross validation scores from 1
MCE = [1 - x for x in crossval_scores]

#determinin best k based on minimum MSE and the trend in the misclassification error wrt number of neighbors
#min(MSE)

optimal_k = parametercount[MCE.index(min(MCE))]

print "the optimal number of neighbors is %d" % optimal_k

plt.plot(parametercount, MCE)
plt.xlabel('Number of Neighbors')
plt.ylabel('Missclassifcation Error')

plt.grid()
plt.show()

# Notes from the test size = 0.2, cv = 10
# from the plot the optimal neighbor count seems to be  1.
# There is a gradual increase in missclassifciation error wrt neighbor count
# there is a decrease in misclassifciation error at 4, however general slope is positive





