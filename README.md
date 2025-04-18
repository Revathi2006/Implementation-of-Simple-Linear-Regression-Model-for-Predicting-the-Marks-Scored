# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries:pandas,numpy,matplotlib,sklearn.
2. Load Dataset:Read CSV file containing study hours and marks
3. Check Data:Preview data and check  for missing values.
4. Define Variables:Set x =Hours,y=Scores.
5. Split Data:Train-test split(80-20).
6. Train Model:Fit Linear Regression on training data.
7. Predict:Use model to predict scores on test data.
8. Evaluate:Calculate Mean Absolute Error(MAE) and R^2 score.
9. Visualize:Plot actual data and regression line.
   

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Revathi.S
RegisterNumber:  212224230228
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score
data =pd.read_csv("/content/student_scores (1).csv")
print("Dataset Preview:\n",data.head())
print("\nMissing Values:\n",data.isnull().sum())
x=data[['Hours']]
y=data[['Scores']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("\nIntercept:",model.intercept_)
print("Slope:",model.coef_[0])
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("\nMean Absolute Error:",mae)
print("R^2 Score:",r2)
plt.scatter(x,y,color='blue',label="Actual Data")
plt.plot(x,model.predict(x),color='red',label='Regression-Line')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression - Marks Prediction")
plt.legend()
plt.show()


```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

![Screenshot 2025-04-18 132227](https://github.com/user-attachments/assets/2ad048f8-822c-45e7-95cd-624a0abccfc5)

![Screenshot 2025-04-18 132245](https://github.com/user-attachments/assets/4af52c25-1c6f-4d48-8d9b-2cf864ad3f9f)


![Screenshot 2025-04-18 132305](https://github.com/user-attachments/assets/3ecf2745-6be9-4d93-bdbf-4d6cfe6b38bd)


![Screenshot 2025-04-18 140110](https://github.com/user-attachments/assets/0cd8ffdf-05da-47a7-b1e1-b413fa4416bd)


![Screenshot 2025-04-18 140145](https://github.com/user-attachments/assets/1219d117-711f-43b3-ae50-dce4cc1a9a8c)









## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
