import numpy as np
import pandas as pd 
import pickle

from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier  

#Import data from csv
train_data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv', sep=",")

#**************** CLEAN DATA *******************
#Eliminating irrelevant data, not use
drop_cols = ['PassengerId','Cabin', 'Ticket', 'Name']
train_data.drop(drop_cols, axis = 1, inplace = True)

#Dropping data with fare above 300
train_data.drop(train_data[(train_data['Fare'] > 300)].index, inplace=True)

# Handling Missing Values in train_data

## Fill missing AGE with Median of the survided and not survided is the same
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

## Fill missing EMBARKED with Mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Encoding

# Encoding the 'Sex' column
train_data['Sex'] = train_data['Sex'].apply(lambda x: 1 if x == 'male' else 0)

# Encoding the 'Embarked' column
train_data['Embarked'] = train_data['Embarked'].map({'S' : 0, 'C': 1, 'Q': 2})

scaler = MinMaxScaler()
train_scaler = scaler.fit(train_data[['Age','Fare']])
train_data[['Age','Fare']] = train_scaler.transform(train_data[['Age','Fare']])

#***************** MODEL *******************

X = train_data[list(train_data.columns[1:9])]
y = train_data[['Survived']]
#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=34)

params = {'criterion': 'entropy',
    'max_depth': 10,
    'max_features': 6,
    'n_estimators': 50}

modelo = RandomForestClassifier(
                oob_score    = True,
                n_jobs       = -1,
                random_state = 123,
                ** params
             )

modelo.fit(X_train, y_train.values.ravel())

#save the model to file
filename = 'models/finalized_model.sav' #use absolute path
pickle.dump(modelo, open(filename, 'wb'))
 