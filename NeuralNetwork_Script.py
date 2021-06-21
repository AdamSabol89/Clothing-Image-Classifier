# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 22:32:00 2021

@author: Localadmin
"""

import pandas as pd
import numpy as np 
import os.path
import random
import cv2
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf 
import keras 
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from keras.applications.vgg16 import VGG16 

#Read in the CSV data file
image_data = pd.read_csv("C:/Users/Localadmin/Clothing Classifier/images.csv")
image_data.head()

#Remove kids clothing
image_data = image_data[image_data['kids'] == False]

#Get only classifications with N>100~
image_data['label'].value_counts()
wanted_attrs = list(image_data['label'].value_counts()[0:15].index)
wanted_attrs.remove('Not sure')

#Get not sure data
not_sure_df = image_data[image_data['label'] == 'Not sure']

#Get rest of data
image_data = image_data[image_data['label'].isin(wanted_attrs)]



#Verify above function
image_data['label'].value_counts()
#We need to balance our dataset 
#First we shuffle the dataset
shuffled_df = image_data.sample(frac=1,random_state=4)

#create an empty dataset
balanced_df = pd.DataFrame()

#iterate through dataframe 
for attribute in wanted_attrs:
    attr_df = shuffled_df.loc[shuffled_df['label'] == attribute].sample(n=97,random_state=42)
    balanced_df = pd.concat([balanced_df, attr_df])

balanced_df['label'].value_counts()

#Before and after
#Before
plt.figure(figsize=(12, 12))
sns.countplot('label', data=shuffled_df)
plt.xticks(rotation=45);
plt.title('Unbalanced Classes')
plt.show()

#After
plt.figure(figsize=(12, 12))
sns.countplot('label', data=balanced_df)
plt.xticks(rotation=45);
plt.title('Balanced Classes')
plt.show()

#Split data into train and test 
y = balanced_df['label'].reset_index(drop=True)
X = balanced_df['image'].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=123)


#Create empty list to append img info to 
X_test_img = []
X_train_img = []

#Set constant img size
IMG_SIZE = 128 

#Get img train data
for i in X_train: 
    base_path = "C:/Users/Localadmin/Clothing Classifier/raw_images/"
    path = base_path + i + ".jpg"
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    print(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X_train_img.append(img)

for i in X_test: 
    base_path = "C:/Users/Localadmin/Clothing Classifier/raw_images/"
    path = base_path + i + ".jpg"
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    print(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X_test_img.append(img)

#Convert to np array        
X_test_img = np.array(X_test_img)
X_train_img = np.array(X_train_img)

plot_img = X_test_img 

#Pre-process images according to vgg16 
X_test_img = tf.keras.applications.vgg16.preprocess_input(X_test_img)
X_train_img = tf.keras.applications.vgg16.preprocess_input(X_train_img)

#Scale data
X_train_img, X_test_img = X_train_img/255, X_test_img/255

#Encode our labels
le = preprocessing.LabelEncoder()
le.fit(y_test)
encoded_y_test = le.transform(y_test)
le.fit(y_train)
encoded_y_train = le.transform(y_train)

#Now we'll start modeling with our own CNN 
INPUT_SHAPE = (128,128,3)

model = keras.Sequential([ 
   layers.Conv2D(32, 3, activation = 'relu', kernel_initializer = 'he_uniform', input_shape = INPUT_SHAPE),
   layers.BatchNormalization(),
   
   layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'),
   layers.Dropout(rate=.1),
   layers.BatchNormalization(),
   layers.MaxPooling2D(),
   
   layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'),
   layers.Dropout(rate=.1),
   layers.BatchNormalization(),
   layers.MaxPooling2D(),
   
   layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'),
   layers.Dropout(rate=.1),
   layers.BatchNormalization(),
   layers.MaxPooling2D(),
   
   layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform'),
   layers.Dropout(rate=.1),
   layers.BatchNormalization(),
   layers.MaxPooling2D(),
   
   layers.Flatten(),
   layers.Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'),
   layers.Dropout(rate=.1),
   layers.Dense(256, activation = 'relu', kernel_initializer = 'he_uniform'),
   layers.Dropout(rate=.2),
   layers.Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'),
   layers.Dropout(rate=.1),

   layers.Dense(14, activation = 'softmax')
   

   ])
model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

#We need to one-hot encode our response before training our model 
y_train_onehot = to_categorical(encoded_y_train)
y_test_onehot = to_categorical(encoded_y_test)

#Train the model
fitted = model.fit(X_train_img, y_train_onehot, epochs=50, validation_data= (X_test_img, y_test_onehot))

#Save the model

model.summary()
prediction_NN = model.predict(X_test_img)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)
 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction_NN)
ax = plt.axes()
sns.heatmap(cm, annot=True)
ax.set_title('Conv Neural Net Confusion Matrix Heatmap')
plt.show()

y_test.reset_index(drop=True, inplace=True) 

n=np.random.randint(0,135+1)
print(n)
img= plot_img[n]
plt.imshow(img)
plt.title("Actual: " + y_test[n] + " | Predicted: " + prediction_NN[n])
print("Actual: " + y_test[n])
print("Predicted: " + prediction_NN[n])

#Plot multiple images
n1,n2,n3 = random.sample(range(1,136), 3)

plt.figure(figsize=(15,10))

img= plot_img[n1]
plt.subplot(231)
plt.imshow(img)
plt.title("Actual: " + y_test[n1] + " | Predicted: " + prediction_NN[n1])

img=plot_img[n2]
plt.subplot(232)
plt.imshow(img)
plt.title("Actual: " + y_test[n2] + " | Predicted: " + prediction_NN[n2])

img=plot_img[n3]
plt.subplot(233)
plt.imshow(img)
plt.title("Actual: " + y_test[n3] + " | Predicted: " + prediction_NN[n3])

#Look's at loss
loss = fitted.history['loss']
val_loss = fitted.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show() 

acc = fitted.history['accuracy']
val_acc = fitted.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#Let's how our data predict's not sure
not_sure_df = not_sure_df['image'].reset_index(drop=True)

not_sure_img = []
for i in not_sure_df: 
    base_path = "C:/Users/Localadmin/Clothing Classifier/raw_images/"
    path = base_path + i + ".jpg"
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    print(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    not_sure_img.append(img)

not_sure_img = np.array(not_sure_img)
plot_img2 = not_sure_img

not_sure_img = tf.keras.applications.vgg16.preprocess_input(not_sure_img)
not_sure_img = not_sure_img/255

prediction_NN_ns = model.predict(not_sure_img)
prediction_NN_ns = np.argmax(prediction_NN_ns, axis=-1)
prediction_NN_ns = le.inverse_transform(prediction_NN_ns)

n=np.random.randint(0,220+1)
img= plot_img2[n]
plt.imshow(img)
plt.title("NN Prediction: " +  prediction_NN_ns[n])

y_test = np.array(y_test)
n = random.sample(range(1,136), 9)
plt.figure(figsize=(13,8))
plt.suptitle('Convolutional Neural Network Predictions', fontsize=16)
for i in range(len(n)):
    img = plot_img[n[i]]
    plt.subplot(int("33" + str(i+1)))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Actual: " + y_test[n[i]] + " | Predicted: " + prediction_NN[n[i]])
 
