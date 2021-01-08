#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 01:42:07 2020

@author: aycaburcu

"""
from keras.preprocessing import image
from keras import models 
import matplotlib.pyplot as plt
import numpy as np
import cv2 

list_name=['/beaver','/boy','/forest','/oak_tree','/snail','/sunflower']
model=models.load_model('cifar100.h5')
test_dir='/mnt/sdb2/ders/deep_learning_bsm/DeepLearning/Cifar100/test'


for x,name in enumerate(list_name):
    tahmin_list=[]
    random=np.random.randint(1,100)
    path=(test_dir+name+'/'+str(random)+".png")
    Giris1=image.load_img(path,
                          target_size=(32,32))

    #Numpy dizisine dönüştür
    Giris=image.img_to_array(Giris1)
    #Görüntüuü ağa uygula
    y=model.predict(Giris.reshape(1,32,32,3))
    #En yüksek tahmin sınıfını bul
    tahmin_indeks=np.argmax(y)
    tahmin_yuzde=y[0][tahmin_indeks]*100
    
    
    ax3 =plt.subplot(6,6,x+1)
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_xlabel('label:{0}\npred:{1}'.format(list_name[x][1:],list_name[tahmin_indeks][1:]))
    plt.imshow(Giris1)
    
    

