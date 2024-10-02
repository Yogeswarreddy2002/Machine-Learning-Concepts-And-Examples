import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp 
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("C:/Users/yogeswar/Desktop/ML/Polynomial regression/Position_Salaries.csv")
x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values

RF = RandomForestRegressor(n_estimators=10,random_state=0)
RF.fit(x, y)
RF.predict([[6.5]])

x_grid = np.arange(min(x),max(x),0.01 )
x_grid = x_grid.reshape(len(x_grid),1)

#visualising
mtp.scatter(x, y)
mtp.plot(x_grid,RF.predict(x_grid))
mtp.title('Truth or Bluff (Random Forest Regression)')
mtp.xlabel('Position level')
mtp.ylabel("salary")
mtp.show()
