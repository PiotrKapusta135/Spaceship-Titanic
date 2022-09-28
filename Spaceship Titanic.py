# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 21:24:15 2022

@author: Acer
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import re

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
import sklearn.svm

train = pd.read_csv(
    'D:/Users/Acer/Python Projects/Python Projects/Data/train.csv')
test = pd.read_csv(
    'D:/Users/Acer/Python Projects/Python Projects/Data/test.csv')

print(train.head())
print(train.info())
print(train.describe())
print('---------------------------------------------------')
print(test.info())
print(train.describe())
print()
print('Missing values for "Train" table:')
for column in train.columns:
    print('Number of missing values in column', column, ': ',
          train[column].isna().sum(),
          round(train[column].isna().sum()/len(train['PassengerId'])*100,2),'%'
          )
          

print()
print('Missing values for "Test" table:')
for column in test.columns:
    print('Number of missing values in column', column, ': ',
          test[column].isna().sum(),
          round(test[column].isna().sum()/len(test['PassengerId'])*100,2),'%'
          )


# -----------------------------------
# Data preparation
# -----------------------------------

train['Label'] = 'train'
test['Label'] = 'test'

df = pd.concat([train, test])

# Nans by row
df['Null'] = df.isnull().sum(axis=1)
df['Null'].value_counts()
   
df['Group'] = df['PassengerId'].str[:4]
df['Nr_in_group'] = df['PassengerId'].str[5:] # People assigned to one group might be family
df['Nr_in_group'] = df['Nr_in_group'].astype('int')
df = df.drop('PassengerId', axis=1)

df['HomePlanet'].value_counts()

# replacing empty values with most common value
df['HomePlanet'].replace(np.nan, 'Earth', inplace=True) 

# Encoding categorical values to numeric
label_encoder = LabelEncoder()
df['HomePlanet'] = label_encoder.fit_transform(df['HomePlanet'])

df['CryoSleep'].value_counts()
df['CryoSleep'].fillna(False, inplace=True)

df = pd.get_dummies(df, columns=['CryoSleep'], drop_first=True)
ax, fig = plt.subplots(len(df.columns)//3,3, figsize=(30,30))
df.hist(ax=ax)
plt.show()
df['Cabin'].sample()
df['Deck'] = df['Cabin'].str[:1]

def get_num(row):
    num = row.split('/')
    return num

df['Cabin_number'] = df['Cabin'].str.split('/').str[1]
df['Side'] = df['Cabin'].str[-1]
df = df.drop(['Cabin'], axis=1)
label_encoder = LabelEncoder()
df['Deck'] = label_encoder.fit_transform(df['Deck'])
df['Side'] = label_encoder.fit_transform(df['Side'])

df['Side'] = df['Side'].replace(2, 0)
df['Deck'].value_counts()
df['Deck'] = df['Deck'].replace(8, 6)
df['Cabin_number'].replace(np.nan, 5000, inplace=True)
df['Cabin_number'] = df['Cabin_number'].astype('int')

df['Cabin_number_bins'] = pd.cut(df['Cabin_number'], 10,labels=[i for i in range(10)])
df['Cabin_number_bins'] = df['Cabin_number_bins'].astype('int')
df.drop('Cabin_number', axis=1, inplace=True)
df.columns

df['Destination'].unique()
df['Destination'].isna().sum()
df['Destination'] = df['Destination'].fillna(df['Destination'].mode().iloc[0])
df['Destination'].value_counts()
label_enc = LabelEncoder()
df['Destination'] = label_encoder.fit_transform(df['Destination'])

def age_split(row):
    if row <=3 :
        val = 0
    elif row > 3 and row <= 12:
        val = 1
    elif row > 12 and row <= 18:
        val = 2
    elif row > 18 and row <= 60:
        val = 3
    else: 
        val = 4 
    return val

df['Age_grouped'] = df['Age'].apply(age_split)
df['Age_grouped'].value_counts()
df.drop('Age', axis=1, inplace=True)
df['VIP'].value_counts()
label_enc = LabelEncoder()
df['VIP'] = label_enc.fit_transform(df['VIP'])
df['VIP'] = df['VIP'].replace(2, 0)

df['Amenities'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']
minmaxsc = MinMaxScaler()
df['Amenities'] = df['Amenities'].replace(np.nan, 0)
df['Amenities'] = minmaxsc.fit_transform(np.array(df['Amenities']).reshape(-1,1))
df.drop(['RoomService', 'FoodCourt',
       'ShoppingMall', 'Spa', 'VRDeck'], axis=1, inplace=True)
df.drop('Name', axis=1, inplace=True)
df.head()
df.columns

df.drop('Null', axis=1, inplace=True)
df['Amenities'].max()
df.max()
label_enc = LabelEncoder()
df['Label'] =label_enc.fit_transform(df['Label'])
df['Nr_in_group'].value_counts()
df = df.drop('Group', axis=1)

train_df = df[df['Label']==1]
test_df = df[df['Label']==0]
train_df = train_df.drop('Label', axis=1)
test_df = test_df.drop('Label', axis=1)

# -----------------------------------
# Modelling
# -----------------------------------

train_df['Transported'] = train_df['Transported'].astype('int')
X = train_df.drop('Transported', axis=1).values
y = train_df['Transported'].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Logistic Regression
LogisticRegr = LogisticRegression()
LogisticRegr.fit(X_train, y_train)
LogisticRegr.predict(X_test)
LogisticRegr.score(X_train, y_train)

params = {'penalty':['l2'], 'C':np.arange(0, 1, 0.1),
         'n_jobs':[-1], 'l1_ratio':[np.arange(0, 1, 0.1)]}
LogRegGrid = GridSearchCV(LogisticRegr, params)
LogRegGrid.fit(X_train, y_train)
LogRegGrid.best_params_
LogRegGrid.predict(X_test)
LogRegGrid.score(X_test, y_test)

#SVC
SVC = sklearn.svm.SVC()
SVC.fit(X_train, y_train)
SVC.predict(X_test)
SVC.score(X_test, y_test)
