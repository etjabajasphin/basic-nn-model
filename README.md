# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

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

## PROGRAM
### Name:
### Register Number:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('12.08.2024').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df=df.astype({'x':'float'})
df=df.astype({'y':'float'})
df
x=df[['x']].values
y=df[['y']].values
df.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)


import numpy as np
from sklearn.preprocessing import MinMaxScaler
mn=MinMaxScaler()
mn.fit(x_train)
x_train1=mn.transform(x_train)

ai_mind=Sequential([
    Dense(8,activation = 'relu',input_shape=[1]),
    Dense(10,activation='relu'),
    Dense(1)
])

ai_mind.compile(optimizer='adam',loss='mse')
ai_mind.fit(x_train1,y_train,epochs=1000)
loss=ai_mind.evaluate(x_test,y_test, verbose=1)
print(f"Test loss: {loss}")

loss=ai_mind.evaluate(x_test,y_test, verbose=1)
print(f"Test loss: {loss}")

new_input=np.array([[20]],dtype=np.float32)
new_input_scaled=mn.transform(new_input)
prediction=ai_mind.predict(new_input_scaled)
print(f'Predicted Value for the input {new_input[0][0]}: {prediction[0][0]}')
```
## Dataset Information

![image](https://github.com/user-attachments/assets/0b39ae6a-329f-4c65-8eea-eaf83d4d7345)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/6e8de703-fe8d-41d3-939a-eed4c1e518a9)
![image](https://github.com/user-attachments/assets/ce966fbf-1820-4d47-ac20-5580d00328b8)



### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/a19b26dc-ce9c-4b3c-b7a3-01a54702b576)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/3b5dac54-4e05-478d-b1bf-3b8cc5fe9940)


## RESULT

Thus a Neural network for Regression model is Implemented
