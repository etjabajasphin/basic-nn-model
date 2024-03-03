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
### Name:V RAKSHA DHARANIKA
### Register Number:212223230167
```python

import pandas as pd


from sklearn.model_selection import train_test_split


from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


from google.colab import auth
import gspread
from google.auth import default




auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)


worksheet = gc.open('exp 1').sheet1
data = worksheet.get_all_values()



dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})


dataset1.head()


X = dataset1[['Input']].values
y = dataset1[['Output']].values


X


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)


Scaler = MinMaxScaler()


Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(5,activation = 'relu'),
    Dense(4,activation = 'relu'),
    Dense(1)
])
ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(X_train1,y_train,epochs = 4000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[10]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)

```
## Dataset Information
```
Input	Output
1	15
2	20
3	25
4	30
5	35
6	40
7	45
8	50
9	55
10	60
11	65
12	70
13	75
14	80
15	85
16	90
17	95
18	100
19	105
20	110

```
## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot (65)](https://github.com/rakshadharanika/basic-nn-model/assets/149348380/7b23e7a7-993b-4134-90e1-46543653efcb)


### Test Data Root Mean Squared Error

![Screenshot (66)](https://github.com/rakshadharanika/basic-nn-model/assets/149348380/01e98642-42ba-49c9-bdb2-56518d4c75b3)


### New Sample Data Prediction

![Screenshot (67)](https://github.com/rakshadharanika/basic-nn-model/assets/149348380/a5b948f7-8e9a-4118-a899-3531f1019581)


## RESULT

Include your result here
