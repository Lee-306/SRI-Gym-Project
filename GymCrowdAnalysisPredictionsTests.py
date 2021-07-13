# -*- coding: utf-8 -*-
"""
Software Design Masters (Artificial Intelligence), Full-Time 
Athlone Institute of Technology

author: Lee O' Connor
Student ID: A00239789
Email: a00239789@student.ait.ie

Program description
<---------------------------------------------------------->

<---------------------------------------------------------->

"""



import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression



def buildDataStructure():
    #read in gym_data
    gym_data = pd.read_csv("GymData.csv") 
    
    
    #convert temperature column from Farenheit to Celcius
    gym_data.temperature = gym_data.temperature.apply(lambda x: int((x-32)*(5/9)))
    gym_data.rename(columns={"temperature":"temperature_celcius"}, inplace = True)
    
    #Change day of week to 1-7 instead of 0-6 for readability, 1 = Monday, 7 = Sunday
    gym_data.day_of_week = gym_data.day_of_week.apply(lambda x: x+1)
    
    #build independent varables and dependent variable( number of people in the gym)
    crowds_X = gym_data.iloc[:, 3:-1]
    crowds_Y = gym_data.iloc[:, 0:1]
    
    #build train and test data
    crowds_X_train, crowds_X_test, crowds_Y_train, crowds_Y_test = train_test_split(crowds_X, crowds_Y, test_size = 0.20, random_state = 8)
    
    return crowds_X_train, crowds_X_test, crowds_Y_train, crowds_Y_test



if __name__ == '__main__':
    pass
    crowds_X_train, crowds_X_test, crowds_Y_train, crowds_Y_test = buildDataStructure()
    
    #test logistic regression
           
    lin_reg_model = LinearRegression()
    
    
    ## ravel() converts array to shape (n,) to avoid error message
    lin_reg_model.fit(crowds_X_train, crowds_Y_train.values.ravel())
    
    crowds_Y_pred = lin_reg_model.predict(crowds_X_test)
    crowds_Y_pred = crowds_Y_pred.astype(int)
    
    acc = accuracy_score(crowds_Y_test, crowds_Y_pred)
    
    print ("Linear regression accuracy: ",acc)
