# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program and Output:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('drive/MyDrive/Dataset-ML/student_scores.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/c95887ac-428e-44a9-a09f-efd7e296de67)
```
df.tail()
```
![image](https://github.com/user-attachments/assets/93027fe9-8a67-4e25-b785-35dabd0166e7)
```
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values
X,Y
```
![image](https://github.com/user-attachments/assets/897ae139-7d17-4344-ac3d-3870cb6fa9c6)
```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
Y_pred,Y_test
```
![image](https://github.com/user-attachments/assets/7c448ff7-cda3-4a9a-be00-6add60127b8a)
```
plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title('Hours VS Scores (Training set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
![image](https://github.com/user-attachments/assets/c4fc8289-5407-46bb-b977-b2d286ed150a)
```
plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='yellow')
plt.title('Hours VS Scores (Test set)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
![image](https://github.com/user-attachments/assets/0c8cee8f-e886-4c67-a380-8ab0d6fc0bda)
```
mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
print("MSE = ",mse)
print('MAE = ',mae)
print('RMSE = ',rmse)
```
![image](https://github.com/user-attachments/assets/4e8522eb-e1c4-432f-a77c-bc7cc99992bd)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
