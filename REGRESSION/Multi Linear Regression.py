import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd 

#importing datasets  
data_set= pd.read_csv('C:/Users/yogeswar/Desktop/ML/Multi linear regression/50_Startups.csv') 

#Extracting Independent and dependent Variable  
x= data_set.iloc[:, :-1].values  
y= data_set.iloc[:, 4].values  

#Catgorical data  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder  
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x = nm.array(ct.fit_transform(x))
#avoiding the dummy variable trap:  
x = x[:, 1:]  
#Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0) 
#step2
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)  
#step3:
#Predicting the Test set result;  
y_pred= regressor.predict(x_test)  
#score:
print('Train Score: ', regressor.score(x_train, y_train))  
print('Test Score: ', regressor.score(x_test, y_test))

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)