# Developing a Neural Network Regression Model
### Name:JAVITH M
### Reference No: 212222110014
## AIM:
To develop a neural network regression model for the given dataset.

## THEORY:
Neural network regression is a supervised learning method, and therefore requires a tagged dataset, which includes a label column. Because a regression model predicts a numerical value, the label column must be a numerical data type. A neural network regression model uses interconnected layers of artificial neurons to learn the mapping between input features and a continuous target variable. It leverages activation functions like ReLU to capture non-linear relationships beyond simple linear trends. Training involves minimizing the loss function (e.g., Mean Squared Error) through an optimizer (e.g., Gradient Descent). Regularization techniques like L1/L2 and dropout prevent overfitting. This approach offers flexibility and high accuracy for complex regression problems.

## Neural Network Model:
![308100610-473c7db9-e1c3-4770-9671-0acddcb30017](https://github.com/Afsarjumail/basic-nn-model/assets/118343395/ae7ed9fe-b1d6-4e08-b979-d2b74dc9b28a)


## DESIGN STEPS:

### STEP 1:
Loading the dataset
### STEP 2:
Split the dataset into training and testing
### STEP 3:
Create MinMaxScalar objects ,fit the model and transform the data.
### STEP 4:
Build the Neural Network Model and compile the model.
### STEP 5:
Train the model with the training data.
### STEP 6:
Plot the performance plot
### STEP 7:
Evaluate the model with the testing data.

## PROGRAM:
### Name: JAVITH M
### Reference No: 2122220014
```
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('exp no 1').sheet1
data=worksheet.get_all_values()
print(data)

dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})

X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.30,random_state = 30)

Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

ai_model=Sequential([
    Dense(units=8,activation='relu',input_shape=[1]),
    Dense(units=9,activation='relu'),
    Dense(units=1)
])

ai_model.compile(optimizer='rmsprop',loss='mse')

ai_model.fit(X_train1,y_train,epochs=20)

loss_df = pd.DataFrame(ai_model.history.history)
loss_df.plot()

X_test1 = Scaler.transform(X_test)
ai_model.evaluate(X_test1,y_test)

X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_model.predict(X_n1_1)
```
## Dataset Information:
![Screenshot 2024-02-27 184025](https://github.com/Afsarjumail/basic-nn-model/assets/118343395/0346746b-39f7-446e-8312-b4a4a0650e93)

## OUTPUT:
### Training Loss Vs Iteration Plot
![308101397-34d24f51-38af-43e0-8b9a-0db401c2d74e](https://github.com/Afsarjumail/basic-nn-model/assets/118343395/8d745483-0249-4e1b-be94-faf2d0c64829)

### Test Data Root Mean Squared Error
![308101895-431be646-1d75-46e7-8b8b-c1101a989262](https://github.com/Afsarjumail/basic-nn-model/assets/118343395/713fa324-643f-465d-8c40-056bac9c6380)

### New Sample Data Prediction
![308101962-e3675586-7dcb-415d-8db3-82bcdf6cba49](https://github.com/Afsarjumail/basic-nn-model/assets/118343395/7ac908b2-22f7-4a3f-97ee-736b12aea385)

## RESULT
A neural network regression model for the given dataset has been developed successfully.
