import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('ex1data1.txt', header=None)             # read from dataset
X = dataset.iloc[:, 0].values                                        # read first column, automatically as a numpy array
y = dataset.iloc[:, 1].values


Xtr,Xte,Ytr,Yte=train_test_split(X, y, test_size=0.25, random_state=0)

regressor=LinearRegression()
Xtr1=Xtr.reshape(1, -1)
Ytr1=Ytr.reshape(1, -1)
regressor.fit(Xtr1,Ytr1)
plt.scatter(Xtr1, Ytr1)
plt.title('X versus Y')
plt.show()
plt.scatter(Xtr, Ytr)
plt.title('X versus Y')
plt.show()
plt.scatter(X, y)
plt.title('X versus Y')
plt.show()



Xte1=Xte.reshape(1, -1)
prediction=regressor.predict(Xte.reshape(-1,1))
print(prediction)
plt.scatter(Xtr, Ytr)

plt.plot(Xtr,  regressor.predict(Xtr), color='red')
plt.title('X versus Y')
plt.show()
