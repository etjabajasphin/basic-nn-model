# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model
![dl1](https://github.com/etjabajasphin/basic-nn-model/assets/118541549/c0f27290-c51a-4906-babc-cab26b2fcb9f)


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
### Name: Rama E.K. Lekshmi
### Register Number: 212222240082
```
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('MARKSDATA').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'int'})
df = df.astype({'OUTPUT':'int'})
df.head()

X = df[['INPUT']].values
y = df[['OUTPUT']].values

X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

marks_data = Sequential([Dense(6,activation='relu'),Dense(7,activation='relu'),Dense(1)])

marks_data.compile(optimizer = 'rmsprop' , loss = 'mse')

marks_data.fit(X_train1 , y_train,epochs = 500)

loss_df = pd.DataFrame(marks_data.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

marks_data.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

marks_data.predict(X_n1_1)
```
## Dataset Information
![Screenshot 2024-02-24 234422](https://github.com/etjabajasphin/basic-nn-model/assets/118541549/a4fde225-e83c-449b-bd94-90cd39fa75a9)


## OUTPUT

### Training Loss Vs Iteration Plot
![dl1 1](https://github.com/etjabajasphin/basic-nn-model/assets/118541549/d2fa8b61-e4ac-41cd-a488-722c8b991e9d)


### Test Data Root Mean Squared Error
![dl 2](https://github.com/etjabajasphin/basic-nn-model/assets/118541549/ded4fc35-4e26-4dae-8990-c9af2c11a7b3)


### New Sample Data Prediction
![dl1 3](https://github.com/etjabajasphin/basic-nn-model/assets/118541549/cab3ca01-b63d-4c6c-8f22-daa406625a0f)


## RESULT
Thus,A Neural Network Regression model for the given dataset has been developed sucessfully.
