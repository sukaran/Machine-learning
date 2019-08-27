import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Dataset.csv')
data.fillna(method ='ffill', inplace = True)
x  = data.iloc[:,-3:-2].values
y  = data.iloc[:,-2:-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state = 0)
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train,y_train)
print(regr.score(x_test, y_test))
y_pred = regr.predict(x_test)
plt.scatter(x_test,y_test,color ='b')
plt.plot(x_test,y_pred,color = 'k')
plt.show()
