# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:24:57 2020

@author: INE12363221
"""

#LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

#PART:01 :DATA PROCESSING 
df=pd.read_csv('iris.csv')
print(df.head())
#df['species_label'], _=pd.factorize(df['species'])
#factorize method is used to get numerical values for categorical values 
#y = df['species_label']
y = pd.get_dummies(df['species']).values
x = df[['petal_length','petal_width','sepal_length','sepal_width']]
print(x.head())
print(y[0])

#STANDARD SCALING 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x= sc.fit_transform(x)

#SPLIT INTO TRAINING AND TEST
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=42 )

#


#MAKE A DEEP LEARNING MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
deep_model = Sequential()
deep_model.add(Dense(32, input_shape=(x.shape[1],), activation='relu'))
deep_model.add(Dense(16, input_shape=(x.shape[1],), activation='relu'))
deep_model.add(Dense(3, activation='softmax'))

deep_model.compile(optimizer = 'Adamax', loss='categorical_crossentropy', metrics=['accuracy'])
deep_model.fit(x_train, y_train, epochs=100, verbose=0)

y_pred=deep_model.predict(x_test)
y_pred_class = deep_model.predict_classes(x_test, verbose=0)
y_test_class = np.argmax(y_test, axis=1)
print(y_pred_class)
print(y_test_class)


#CHECKING ACCURACY 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test_class,y_pred_class)
print('Accuracy: {:.2f}'.format(accuracy))
#How many we have mis classified 
count_misclassified = (y_test_class != y_pred_class).sum()
print('Misclassified samples: {}'.format(count_misclassified))

#PREDICTING INDIVIDUAL ITEM 
print("making individual prediction")
new_data = np.array([[5.1,3.5,1.4,0.2]])    
new_data_scaled = sc.transform(new_data)
res=deep_model.predict_classes(new_data_scaled)

perclass=deep_model.predict_proba(new_data_scaled)
probabilities=perclass.flatten()
types=[0,1,2]
for i in range( 0,len(types)):
    print(str(types[i])+" has probability "+str(round(probabilities[i],4)))
print("So selecting result ",res)  

out="model failed "
#MAP IT TO NAME OF FLOWER
if res==0:
    out="SETOSA"
elif res==1:
    out="VERSICOLOR"
elif res==2:
    out="VIRGINICA"
print(out)  

