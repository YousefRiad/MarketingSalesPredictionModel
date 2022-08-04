import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt





#loading and storing Dateframe:
df = pd.read_csv('Marketing_Data.csv')

#Assigning input features to x:
x = df.drop(['sales'],axis=1).values

#Assigning Outputs to y:
y = df['sales'].values

#splitting Dataframe using "Train_Test_Split" function (70% Train, 30% Test):
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#we use sklearn linear regression function to train the model:
rg = LinearRegression()
rg.fit(x_train,y_train)

#we use x_test to make predictions then store them in y_hat:
y_hat = rg.predict(x_test)


#testing our model with actual input features from the training set for comparison:
#note: if model was well trained the results should be very close to the actual y values.
rg.predict([[278.52,10.32,10.44]])

#using the sklearn "r2_score" we test the accuracy of our predicted values (y_hat) VS actual values (y_test):
r2_score (y_test,y_hat)

#ploting Actual vs Predicted:
plt.figure(figsize=(15,10))
plt.scatter(y_test,y_hat)
plt.xlabel('Actual',fontsize=15)
plt.ylabel('Predicted',fontsize=15)
plt.title('Actual vs Predicted')


#calculating the difference between Acual and Predicted values and putting them side by side:
cost = pd.DataFrame({'Actual':y_test,'Predicted':y_hat,'Difference': y_test - y_hat})


#storing the recently created "Actual vs Predicted" Dataframe in a new csv file:
cost.to_csv("sales_predictions.csv",index = False)
