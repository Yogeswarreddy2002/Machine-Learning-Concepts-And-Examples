import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd

df = pd.read_csv("C:/Users/yogeswar/Desktop/ML/Polynomial regression/Position_Salaries.csv")
x = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values
y = y.reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel="rbf")
regressor.fit(x, y)

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

mtp.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y),color = "red")
mtp.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)),color = "green")
mtp.title('Truth or Bluff (SVR)')
mtp.xlabel('Position level')
mtp.ylabel('Salary')
mtp.show()

x_grid = np.arange(min(sc_x.inverse_transform(x)), max(sc_x.inverse_transform(x)), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
mtp.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
mtp.plot(x_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(x_grid)).reshape(-1,1)), color = 'blue')
mtp.title('Truth or Bluff (SVR)')
mtp.xlabel('Position level')
mtp.ylabel('Salary')
mtp.show()

