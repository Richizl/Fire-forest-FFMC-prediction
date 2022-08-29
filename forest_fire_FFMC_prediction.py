
"""
"Forest Fire, FFMC prediction"
Ricardo Zepeda - A01174203
29.08.2022
"""

#from google.colab import drive

#drive.mount("/content/gdrive")  
#!pwd


# Commented out IPython magic to ensure Python compatibility.
# %cd "/content/gdrive/MyDrive/Intelligent Systems/Project 1 Fire prediction"
#!ls

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#Read dataset.
df=pd.read_csv('forestfires.csv')
df.head()


#Selecting corelated features
pd.plotting.scatter_matrix(df[['FFMC', 'temp', 'RH', 'wind', 'rain']]);

#Temperature and FFMC show the closest linear relation)
df.plot('temp', 'FFMC', kind ='scatter')

# Separate features and labels
X = df[["temp"]] #Temperature
y = df[["FFMC"]] #he Fine Fuel Moisture Code (FFMC) denotes the moisture content surface litter and influences ignition and fire spread.

#Separate dataset into training and tesing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print('Mean squared error: %.2f'
% mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
% r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()

#Prediction interface
Predict = input("\n To make a FMMC prediction press 1: ")

while Predict == '1':
    Temp = input("\nEnter the Ambient Temperature: ")
    features = np.expand_dims(np.array([Temp], dtype = 'float64'), axis=0)
    
    # Perform prediction
    new_prediction = regr.predict(features)
    print("\nThe expected Fine Fuel Moisture Code (FFMC) is: ", new_prediction)

    Predict = input("\nTo make another prediction press 1, if not press 0: ")