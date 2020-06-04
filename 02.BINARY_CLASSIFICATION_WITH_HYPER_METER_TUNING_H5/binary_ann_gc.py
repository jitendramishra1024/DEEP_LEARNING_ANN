# -*- coding: utf-8 -*-
"""BINARY_ANN_GC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qeimLq8JDIiSYvCTpVUocCMCKa48fTGu
"""

!pip install -q keras
#installation of keras

#Read the file from drive 
import pandas as pd 
dataset=pd.read_csv('/content/drive/My Drive/Colab Notebooks/Churn_Modelling.csv')
dataset.info()

#DATA PRE PROCESSING 
#STEP 1: 
#CHECK NULL INFO 
#now we came to know surname geography and gender are of Object type 
print(dataset.isnull().sum(axis=0))

#DATA PREPARATION 
#As we know Rownumber ,customer ID and surname not going to make any impact on prediction 
x=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]
x=pd.get_dummies(x,drop_first=True)

#SPLIT DATA INTO TRAIN AND TEST 
#splitting the dataset into the training set and test set 
x=x.values
y=y.values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2,random_state=42 )

#FEATURE SCALING 
#Feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#LETS MAKE ANN
import tensorflow.keras 
import tensorflow.keras 
#Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. 
from tensorflow.keras.models import Sequential
#The Sequential model is a linear stack of layers.
from tensorflow.keras.layers import Dense
#Dense implements the operation: output = activation(dot(input, kernel) + bias) 
#where activation is the element-wise activation function passed as the activation argument, 
#kernel is a weights matrix created by the layer, and bias is a bias vector created by the layer (only applicable if use_bias is True).
from tensorflow.keras.layers import LeakyReLU,PReLU,ELU
from tensorflow.keras.layers import Dropout
#Dropout consists in randomly setting a fraction rate of input units to 0 at each update during training time, which helps prevent overfitting.

classifier=Sequential()
#Adding input layer
classifier.add(Dense(units=10, input_shape=(11,),kernel_initializer='he_uniform'))
classifier.add(Dropout(0.3))
#NUMBER OF NODES IN INPUT LAYER :11 as number of column is 11 in input feature OUTPUT will be 6 which will be input to hidden layer 

# Adding the First hidden layer
classifier.add(Dense(units=20, kernel_initializer = 'he_uniform',activation='relu'))
classifier.add(Dropout(0.4))
#Adding second hidden layer 
classifier.add(Dense(units=10, kernel_initializer = 'he_normal',activation='relu'))
classifier.add(Dropout(0.2))
#Adding third hidden layer 
classifier.add(Dense(units=6, kernel_initializer = 'he_normal',activation='relu'))
classifier.add(Dropout(0.2))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model=classifier.fit(x_train, y_train,validation_split=0.33, batch_size = 10, epochs = 50)

import matplotlib.pyplot as plt 
# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Predicting the Test set results
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

!pip install scikit-learn==0.21.2

from sklearn import metrics
print('AUC: %.3f' % metrics.roc_auc_score(y_test, y_pred))
#ACCURACY
print("Accuracy score",metrics.accuracy_score(y_test, y_pred))
#F1score 
print("F1_score",metrics.f1_score(y_test, y_pred, average='macro'))
#Precision  
print("precision_score",metrics.precision_score(y_test, y_pred))
#Recall
print("Recall_score",metrics.recall_score(y_test, y_pred))

#CONFUSION MATRIX
print(" Cofusion matrix score")
print(metrics.confusion_matrix(y_test, y_pred))

#LETS DO HYPER METER TUNING

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#To do cross-validation with keras we will use the wrappers for the Scikit-Learn API
#There are two wrappers available:
#keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params), which implements the Scikit-Learn classifier interface,
#keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params), which implements the Scikit-Learn regressor interface.

#SOLUTION IS GRID  SEARCH 
def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_shape=(x_train.shape[1],)))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=1)

layers = [[20], [10, 5], [15, 10, 5]]
#First try with single hidden layer with 20 neurons #2nd try 2 layers first with 40 neuron second with 20 neuron 
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=3)

grid_result = grid.fit(x_train, y_train)

#print best parameters 
print("Best params obtained ",grid_result.best_params_)
print("Best score obtained ",grid_result.best_score_)

# Predicting the Test set results
y_pred = grid.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn import metrics
print('AUC: %.3f' % metrics.roc_auc_score(y_test, y_pred))
#ACCURACY
print("Accuracy score",metrics.accuracy_score(y_test, y_pred))
#F1score 
print("F1_score",metrics.f1_score(y_test, y_pred, average='macro'))
#Precision  
print("precision_score",metrics.precision_score(y_test, y_pred))
#Recall
print("Recall_score",metrics.recall_score(y_test, y_pred))

#CONFUSION MATRIX
print(" Cofusion matrix score")
print(metrics.confusion_matrix(y_test, y_pred))

#Saving Model
#Save Your Neural Network Model to JSON
#The Hierarchical Data Format (HDF5) is a data storage format for storing large
#arrays of data including values for the weights in a neural network.
#You can install HDF5 Python module: pip install h5py
#Keras gives you the ability to describe and save any model using the JSON format.

!pip install h5py

#h5 file stores model and architecture together

#Saving model 
classifier.save("/content/drive/My Drive/Colab Notebooks/BINARY_CLF_ANN.h5")
#Reading model
from tensorflow.python.keras.models import load_model 
# load model
loaded_model = load_model("/content/drive/My Drive/Colab Notebooks/BINARY_CLF_ANN.h5")

# Predicting the Test set results
y_pred = loaded_model.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn import metrics
print('AUC: %.3f' % metrics.roc_auc_score(y_test, y_pred))
#ACCURACY
print("Accuracy score",metrics.accuracy_score(y_test, y_pred))
#F1score 
print("F1_score",metrics.f1_score(y_test, y_pred, average='macro'))
#Precision  
print("precision_score",metrics.precision_score(y_test, y_pred))
#Recall
print("Recall_score",metrics.recall_score(y_test, y_pred))

#CONFUSION MATRIX
print(" Cofusion matrix score")
print(metrics.confusion_matrix(y_test, y_pred))

#Same as model trained earlier