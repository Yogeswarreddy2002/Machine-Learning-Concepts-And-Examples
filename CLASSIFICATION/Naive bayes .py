import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
import seaborn as sns

df = pd.read_csv('C:/Users/yogeswar/Desktop/ML/CLASSIFICATION/Logistic Regression/Social_Network_Ads.csv')
x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
nb.predict(sc.transform([[30,87000]]))
y_pred= nb.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
AS = accuracy_score(y_test,y_pred)
 


sns.heatmap(cm, annot=True)
