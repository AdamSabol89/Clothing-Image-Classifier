# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 00:55:08 2021

"""
import pandas as pd
import numpy as np 
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
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle 

image_data = pd.read_csv("C:/Users/Localadmin/Clothing Classifier/images.csv")
image_data.head()
image_data = image_data[image_data['kids'] == False]

image_data['label'].value_counts()
wanted_attrs = list(image_data['label'].value_counts()[0:15].index)
wanted_attrs.remove('Not sure')

#Get not sure data
not_sure_df = image_data[image_data['label'] == 'Not sure']

#Get rest of data
image_data = image_data[image_data['label'].isin(wanted_attrs)]


shuffled_df = image_data.sample(frac=1,random_state=4)

#create an empty dataset
balanced_df = pd.DataFrame()

#iterate through dataframe 
for attribute in wanted_attrs:
    attr_df = shuffled_df.loc[shuffled_df['label'] == attribute].sample(n=97,random_state=42)
    balanced_df = pd.concat([balanced_df, attr_df])

#This time we will keep an unbalanced version of the data frame
y = balanced_df['label'].reset_index(drop=True)
X = balanced_df['image'].reset_index(drop=True)

y2 = shuffled_df['label'].to_numpy()
X2 = shuffled_df['image'].to_numpy()

#Split data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=123)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=.1, random_state=123)

#Load in data 
#Create empty list to append img info to 
X_test_img = []
X_train_img = []

X2_test_img = []
X2_train_img = []

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

#Get img train data
for i in X2_train: 
    base_path = "C:/Users/Localadmin/Clothing Classifier/raw_images/"
    path = base_path + i + ".jpg"
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    print(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X2_train_img.append(img)

for i in X2_test: 
    base_path = "C:/Users/Localadmin/Clothing Classifier/raw_images/"
    path = base_path + i + ".jpg"
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    print(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X2_test_img.append(img)

#send to np array
X_test_img = np.array(X_test_img)
X_train_img = np.array(X_train_img)

X2_test_img = np.array(X2_test_img)
X2_train_img = np.array(X2_train_img)

#Save for later plotting 
plot_img = X_test_img 
plot2_img = X2_test_img 


#Pre-process images according to vgg16 
X_test_img = tf.keras.applications.vgg16.preprocess_input(X_test_img)
X_train_img = tf.keras.applications.vgg16.preprocess_input(X_train_img)
X2_test_img = tf.keras.applications.vgg16.preprocess_input(X2_test_img)
X2_train_img = tf.keras.applications.vgg16.preprocess_input(X2_train_img)

#Scale our data
X_train_img, X_test_img = X_train_img/255, X_test_img/255
X2_train_img, X2_test_img = X2_train_img/255, X2_test_img/255

#Loag in VGG model
VGG_model = VGG16(weights= 'imagenet', include_top=False, input_shape =(IMG_SIZE, IMG_SIZE,3 ))

VGG_model.summary()
#Disable training for layers
for layer in VGG_model.layers: 
    layer.trainable=False

#get convolutional layers
features_X1 = VGG_model.predict(X_train_img) 
features_X2 = VGG_model.predict(X2_train_img)
features_X1_test = VGG_model.predict(X_test_img)
features_X2_test = VGG_model.predict(X2_test_img)

#get shape for XGBoost
features_X1_train = features_X1.reshape(features_X1.shape[0], -1)
features_X2_train = features_X2.reshape(features_X2.shape[0], -1)
features_X1_test = features_X1_test.reshape(features_X1_test.shape[0], -1)
features_X2_test = features_X2_test.reshape(features_X2_test.shape[0], -1)

#dataset 1 
model = XGBClassifier()
model.fit(features_X1_train, y_train, verbose=True)
y_pred = model.predict(features_X1_test)
print("Accuracy: " + str(accuracy_score(y_test, y_pred)))
print("Precision: " + str(precision_score(y_test, y_pred, average = 'macro'))) 
print("Recall: " + str(recall_score(y_test, y_pred, average = 'macro')))


#Save the model
#pickle.dump(model, open("balanced_data.pickle.dat", "wb"))
loaded_model = pickle.load(open("balanced_data.pickle.dat", "rb"))
y_pred = loaded_model.predict(features_X1_test)

#make confusion matrix balanced dataset
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
ax = plt.axes()
sns.heatmap(cm, annot=True)
ax.set_title('XGBoost Balanced-Data Confusion Matrix Heatmap')
plt.show()

#dataset 2
model2 = XGBClassifier()
model2.fit(features_X2_train, y2_train)
y2_pred = model2.predict(features_X2_test)
print("Accuracy: " + str(accuracy_score(y2_test, y2_pred)))
print("Precision: " + str(precision_score(y2_test, y2_pred, average = 'weighted'))) 
print("Recall: " + str(recall_score(y2_test, y2_pred, average = 'weighted')))

#Save and load model
pickle.dump(model2, open("full_data.pickle.dat", "wb"))
model2 = pickle.load(open("full_data.pickle.dat", "rb"))


#make confusion matrix full-data
cm = confusion_matrix(y2_test, y2_pred)
ax = plt.axes()
sns.heatmap(cm, annot=True)
ax.set_title('XGBoost Full-Data Confusion Matrix Heatmap')
plt.show()

#Balanced dataset predictions 
n=np.random.randint(0,135+1)
print(n)
img= plot_img[n]
plt.imshow(img)
plt.title("Actual: " + y_test[n] + " | Predicted: " + y_pred[n])
print("Actual: " + y_test[n])
print("Predicted: " + y_pred[n])

#Full dataset predictions 
n=np.random.randint(0,459+1)
print(n)
img= plot2_img[n]
plt.imshow(img)
plt.title("Actual: " + y2_test[n] + " | Predicted: " + y2_pred[n])
print("Actual: " + y2_test[n])
print("Predicted: " + y2_pred[n])

#Predict unsure 
#Load in the data
not_sure_df = not_sure_df['image'].reset_index(drop=True)

not_sure_img = []
for i in not_sure_df: 
    base_path = "C:/Users/Localadmin/Clothing Classifier/raw_images/"
    path = base_path + i + ".jpg"
    img  = cv2.imread(path, cv2.IMREAD_COLOR)
    print(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    not_sure_img.append(img)

#send to array and save data for plotting
not_sure_img = np.array(not_sure_img)
plot_img_ns = not_sure_img

#pre-process the data
not_sure_img = tf.keras.applications.vgg16.preprocess_input(not_sure_img)
not_sure_img = not_sure_img/255

#extract features
features_not_sure = VGG_model.predict(not_sure_img)
features_not_sure = features_not_sure.reshape(features_not_sure.shape[0], -1)

#predict model
not_sure_preds = model.predict(features_not_sure)

n=np.random.randint(0,220+1)
img= plot_img_ns[n]
plt.imshow(img)
plt.title("XGBoost Prediction: " +  not_sure_preds[n])


#Figure Coding 
fig, axs = plt.subplots(ncols=2)
fig.set_figheight(8)
fig.set_figwidth(20)
for ax in axs.flatten():
    for label in ax.get_xticklabels():
        label.set_rotation(45)
axs[0].set_title("Unbalanced Dataset")
axs[1].set_title("Balanced Dataset")
sns.countplot('label', data=shuffled_df, ax=axs[0])
sns.countplot('label', data=balanced_df, ax=axs[1])


n = random.sample(range(1,460), 9)
plt.figure(figsize=(13,8))
plt.suptitle('XGBoost Predictions', fontsize=16)
for i in range(len(n)):
    img = plot2_img[n[i]]
    plt.subplot(int("33" + str(i+1)))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Actual: " + y2_test[n[i]] + " | Predicted: " + y2_pred[n[i]])

    
n = random.sample(range(1,221), 9)
plt.figure(figsize=(13,8))
plt.suptitle('XGBoost \"Not Sure\" Predictions', fontsize=16)
for i in range(len(n)):
    img = plot_img_ns[n[i]]
    plt.subplot(int("33" + str(i+1)))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Predicted: " + not_sure_preds[n[i]])


scores = {}
scores['conv_nn'] = accuracy_score(y_test, prediction_NN)
scores['XGBoost_1'] = accuracy_score(y_test, y_pred) 
scores['XGBoost_2'] = accuracy_score(y2_test, y2_pred)

fig = plt.figure(figsize=(8, 6))
ax = sns.pointplot(x=list(scores.keys()), y=[score for score in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0])
plt.ylabel('Accuracy', size=10, labelpad=12.5)
plt.xlabel('Model', size=14, labelpad=12.5)

