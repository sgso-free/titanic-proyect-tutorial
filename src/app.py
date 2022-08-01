import numpy as np
import pandas as pd 
import pickle
import xgboost as xgb

from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  

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

#***************** MODEL RANDOMFOREST*******************

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

print('Accuracy RandomForest:',modelo.score(X_test, y_test)) 

#save the model to file
filename = 'models/finalized_model.sav' #use absolute path
pickle.dump(modelo, open(filename, 'wb'))
 
#***************** MODEL GRADIENT BOOSTING - sklearn*******************

param_best_GBC =  {'criterion': 'friedman_mse',
            'learning_rate': 0.15,
            'max_depth': 5,
            'max_features': 'sqrt',
            'min_samples_leaf': 0.1,
            'min_samples_split': 0.1,
            'n_estimators': 50}

clf_GBBest = GradientBoostingClassifier(random_state=34,**param_best_GBC) 

clf_GBBest.fit(X_train, y_train.values.ravel()) 

print('Accuracy GB:',clf_GBBest.score(X_test, y_test)) 

#save the model to file
filename = 'models/boosting_model.sav' #use absolute path
pickle.dump(modelo, open(filename, 'wb'))

#***************** MODEL BOOSTING - XGB*******************

 
XB_train = X_train.copy()
XB_test = X_test.copy()
yB_train = y_train.copy()
yB_test = y_test.copy()


XB_train['Sex'] = X_train['Sex'].astype(int)
XB_train['Embarked'] = X_train['Embarked'].astype(int)

XB_test['Sex'] = XB_test['Sex'].astype(int)
XB_test['Embarked'] = XB_test['Embarked'].astype(int)


yB_train['Survived'] = yB_train['Survived'].astype(int)
yB_test['Survived'] = yB_test['Survived'].astype(int)

param_XGB ={'booster': 'gbtree',
 'colsample_bytree': 0.7,
 'eta': 0.3,
 'max_depth': 6,
 'min_child_weight': 3,
 'n_estimators': 20,
 'objective': 'binary:logistic'}

clf_Xgb = xgb.XGBClassifier(random_state=35,**param_XGB)

clf_Xgb.fit(XB_train, yB_train.values.ravel()) 

print('Accuracy XGB:',clf_Xgb.score(XB_test, yB_test)) 

#save the model to file
filename = 'models/xgb_model.sav' #use absolute path
pickle.dump(modelo, open(filename, 'wb'))