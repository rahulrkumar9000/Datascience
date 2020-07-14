# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:31:36 2020

@author: sinkur
"""

import cv2
import os
from PIL import Image
import glob
import numpy as np
from tqdm import tqdm

#Setting up the images and making it standard
X=[]
y=[]
IMG_SIZE=150

#image Directories 
rule9dirtrain='C:/Users/sinkur/Documents/Consulting/NHA/rule9/train/rule9'
nonrule9train_DIR='C:/Users/sinkur/Documents/Consulting/NHA/rule9/train/Nonrule9'
nonrule9test_DIR='C:/Users/sinkur/Documents/Consulting/NHA/rule9/test/Nonrule9'
rule9test_DIR='C:/Users/sinkur/Documents/Consulting/NHA/rule9/test/rule9'

#Assigning labels to images

def assign_label(img,label):
    return label

def make_train_data(label,Dir):
    for img in tqdm(os.listdir(Dir)):
        label=assign_label(img,label)
        path = os.path.join(Dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        X.append(np.array(img))
        y.append(str(label))

make_train_data('rule9',rule9dirtrain)
print(len(X))
make_train_data('rule9',rule9test_DIR)
print(len(X))
make_train_data('nonrule9',nonrule9train_DIR)
print(len(X))
make_train_data('nonrule9',nonrule9test_DIR)
print(len(X))

#Visualizing
import matplotlib.pyplot as plt
import random as rn

fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(y))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('images '+y[l])    
plt.tight_layout()

#Model Building 
#Importing the required libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#Deep Learning libraries
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

#label encoding & standardizing
le=LabelEncoder()
Y=le.fit_transform(y)
Y=to_categorical(Y,2)
X=np.array(X)
X=X/255

#Partitioning the data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
np.random.seed(42)
rn.seed(42)

#Architecting the network
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


#2nd Block
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#3rd Block
model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

#4th Block
model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(rate=0.25))
model.add(Activation('relu'))
model.add(Dense(2, activation = "softmax"))

batch_size=32
epochs=15

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
red_lr= ReduceLROnPlateau(monitor='val_accuracy',patience=3,verbose=1,factor=0.1)
model_checkpoint_callback = ModelCheckpoint(filepath="C:\\Users\\sinkur\\Documents\\Consulting\\NHA\\rule9\\Rule9.h5",save_weights_only=False,monitor='val_accuracy',mode='max',
    save_best_only=True)

mycallbacks=[model_checkpoint_callback]

datagen = ImageDataGenerator(
        featurewise_center=False,  
        samplewise_center=False,  
        featurewise_std_normalization=False,  
        samplewise_std_normalization=False,  
        zca_whitening=False,  
        rotation_range=10,  
        zoom_range = 0.1, 
        width_shift_range=0.2,  
        height_shift_range=0.2,  
        horizontal_flip=True,  
        vertical_flip=False)  


datagen.fit(x_train)

model.summary()

model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size,callbacks
                              =mycallbacks)



plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

#Model Validation
from keras.models import load_model
modelx= load_model("C:\\Users\\sinkur\\Documents\\Consulting\\NHA\\rule9\\Rule9.h5")
pred=modelx.predict(x_test)
pred_digits=np.argmax(pred,axis=1)

preds=np.round(pred,0)

lab=["rule9","nonrule9"]

classification_metrics= classification_report(y_test,preds,target_names=lab)

import pandas as pd
categorical_test_labels = pd.DataFrame(y_test).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1) #columnwise indices for maximum value
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)

#################################Final_test######################################################
z=[]
IMG_SIZE=150
Dir="C:\\Users\\sinkur\\Documents\\Consulting\\NHA\\rule9\\Real_test"
for img in tqdm(os.listdir(Dir)):
        path = os.path.join(Dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        z.append(np.array(img))
    
pred_z= modelx.predict(np.array(z))
pred_digits_z=np.argmax(pred,axis=1)
preds_z=np.round(pred_z,0)
categorical_preds_z = pd.DataFrame(preds_z).idxmax(axis=1)
################################################################################################


# the vgg16 model
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
# load model without classifier layers
model = VGG16(include_top=False, input_shape=(224, 224, 3))
# add new classifier layers
flat1 = Flatten()(model.outputs)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(2, activation='sigmoid')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize
model.summary()






