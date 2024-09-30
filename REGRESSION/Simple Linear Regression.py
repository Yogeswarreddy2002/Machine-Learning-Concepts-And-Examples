#IMPORT LIBRARIES
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
#IMPORT DATASETS
df = pd.read_csv(r'C:\Users\yogeswar\Downloads\Salary_Data.csv')
x = df.iloc[:,: -1].values
y = df.iloc[:,1].values
#iloc will select positions 
#import scikit learn , import train and test split , Assigning x,y variables to test train  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1/3 , random_state= 0)
#import sklearn model and import linear regression class
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#fit method uses to fit objects 
regressor.fit(x_train, y_train)

x_pred = regressor.predict(x_train)


#visualising the train set results
mtp.scatter(x_train, y_train, color = 'green')
mtp.plot(x_train, x_pred,color = "red")
mtp.title("salary vs experinced (Training datasets)")
mtp.xlabel("years of experience")
mtp.ylabel("salary")
mtp.show()

#visualising the test set results
mtp.scatter(x_test, y_test, color = "blue")
mtp.plot(x_train, x_pred, color = "yellow")
mtp.title("salary vs experinced (Training datasets)")
mtp.xlabel("years of experience")
mtp.ylabel("salary")
mtp.show()
