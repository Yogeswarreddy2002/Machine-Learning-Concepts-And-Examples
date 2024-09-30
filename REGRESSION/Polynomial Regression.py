import numpy as np
import pandas as pd
import matplotlib.pyplot as mlt
df = pd.read_csv("C:/Users/yogeswar/Desktop/ML/Position_Salaries.csv")
x = df.iloc[:,1:2].values
y = df.iloc[:,2].values
#linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)

#polynomial regression model

from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree=2)
x_poly = pr.fit_transform(x)
lr2 = LinearRegression()
lr2.fit(x_poly, y)

#visualising the linear regresion
mlt.scatter(x, y,color = "blue")
mlt.plot(x,lr.predict(x), color = "red")
mlt.xlabel("Positivee levels")
mlt.ylabel("salary")
mlt.title("Bluff Detection model(Linear regression)")
mlt.show()
#visualising the polynomial regression model
mlt.scatter(x, y,color = "blue")
mlt.plot(x,lr2.predict(pr.fit_transform(x)), color = "red")
mlt.xlabel("Positivee levels")
mlt.ylabel("salary")
mlt.title("Bluff Detection model(Polynomial regression)")
mlt.show()
#Predicting the final result with the Linear Regression model:
lr_pred= lr.predict([[6.5]])
#Predicting the final result with the Polynomial Regression model:

pr_pred = lr2.predict(pr.fit_transform([[6.5]]))




