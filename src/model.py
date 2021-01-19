#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 00:20:29 2020

@author: aycaburcu
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.datasets import cifar100
from keras import layers
from keras import models
import tensorflow as tf
import numpy as np
import cv2
import os
from keras import optimizers
from tensorflow.keras import activations



train_dir='/mnt/sdb2/ders/deep_learning_bsm/DeepLearning/Cifar100/train'
test_dir='/mnt/sdb2/ders/deep_learning_bsm/DeepLearning/Cifar100/test'


   
# ağı tanımlama
model=models.Sequential()
#1.katman 

model.add(layers.Conv2D(32,(3,3),#(3,3) olarak belirtilen kernel boyutu
                        padding='same',#kenarlara dolgulama işlemi uygulanicak
                        activation='relu',#aktivasyon fonksiyonu relu
                        input_shape=(32,32,3)))#32,32 boyutunda renkli bir görüntü

#2.KATMAN
model.add(layers.Conv2D(32,(3, 3),#(3,3) olarak belirtilen kernel boyutu
                         padding='same',#kenarlara dolgulama işlemi uygulanicak
                         activation='relu'))#aktivasyon fonksiyonu relu

model.add(layers.MaxPooling2D((2, 2)))



#3.KATMAN
model.add(layers.Conv2D(64,(3, 3),#(3,3) olarak belirtilen kernel boyutu
                         padding='same',#kenarlara dolgulama işlemi uygulanicak
                         activation='relu'))#aktivasyon fonksiyonu relu



#4.Katman
model.add(layers.Conv2D(64,(3,3),#(3,3) olarak belirtilen kernel boyutu
                        padding='same',#kenarlara dolgulama işlemi uygulanicak
                        activation='relu'))#aktivasyon fonksiyonu relu
model.add(layers.MaxPooling2D((2, 2)))

#5.Katman
model.add(layers.Conv2D(128,(3,3),#(3,3) olarak belirtilen kernel boyutu
                        padding='same',#kenarlara dolgulama işlemi uygulanicak
                        activation='relu'))#aktivasyon fonksiyonu relu
#6.Katman
model.add(layers.Conv2D(128,(3,3),#(3,3) olarak belirtilen kernel boyutu
                        padding='same',#kenarlara dolgulama işlemi uygulanicak
                        activation='relu'))#aktivasyon fonksiyonu relu






#Dense layer:
model.add(layers.Flatten())
model.add(layers.Dropout(0.50))
model.add(layers.Dense(256, activation='relu'))#aktivasyon fonksiyonu relu
model.add(layers.Dropout(0.50))
model.add(layers.Dense(256, activation='relu'))#aktivasyon fonksiyonu relu
model.add(layers.Dense(6, activation='softmax'))

#modeli compile ettik

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adamax(lr=5e-3), metrics=['acc'])

#dizayn edilen modelin içinde kaç tane parametre var onu görmemizi sağlar
print(model.summary())

train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen= ImageDataGenerator(rescale=1./255)


#veri zenginleştirilmesi yapılıyor overfitting engellemek için
datagen = ImageDataGenerator(
              zoom_range=0.2,
              horizontal_flip=True,
              rescale=1./255)

#Train:
train_generator = datagen.flow_from_directory(train_dir,
                                          target_size = (32,32),
                                          batch_size =20,
                                          class_mode = 'categorical')
 
test_generator = datagen.flow_from_directory(test_dir,
                                          target_size = (32,32),
                                          batch_size = 20,
                                          class_mode = 'categorical')

history = model.fit_generator(train_generator,
                              steps_per_epoch=150,                              
                              epochs=30,
                              validation_data=test_generator,
                              validation_steps=30) 

def plot_acc_loss(x):  
  acc = x.history["acc"]
  val_acc = x.history["val_acc"]
  loss = x.history["loss"]
  val_loss = x.history["val_loss"]
  
  print("acc =", acc[-1])
  print("val_acc = ",val_acc[-1])
  print("loss =", loss[-1])
  print("val_loss =", val_loss[-1])
  epochs = range(1, len(acc) + 1)
  fig=plt.figure()
  plt.subplot(2,1,1)
  plt.plot(epochs, acc, "bo", label="Training acc")
  plt.plot(epochs, val_acc, "b", label="Validation acc")
  plt.xlabel('Epoch')
  plt.ylabel('Acc')
  plt.title("Training and Validation Accuracy")

  plt.subplot(2,1,2)
  plt.plot(epochs, loss, "bo", label="Training loss")
  plt.plot(epochs, val_loss, "b", label="Validation loss")
  plt.title("Training and Validation Loss")
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  fig.tight_layout()
  plt.show()
  


plot_acc_loss(history)
model.save('cifar100.h5')









