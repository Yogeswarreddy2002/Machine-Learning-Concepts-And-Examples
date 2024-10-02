import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

df = pd.read_csv("C:/Users/yogeswar/Desktop/ML/Polynomial regression/Position_Salaries.csv")
x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(random_state=0)
DT.fit(x, y)

DT.predict([[6.5]])

#visualising 

x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))

mtp.scatter(x, y,color = "red")
mtp.plot(x_grid,DT.predict(x_grid), color = 'blue')
mtp.title("Truth or Bluff (Decision Tree Regression)")
mtp.xlabel('Position level')
mtp.ylabel('Salary')
mtp.show()
