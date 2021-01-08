#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:42:58 2020

@author: aycaburcu
"""







import json
from matplotlib import pyplot
import cv2 
import numpy as np
import os
import tensorflow as tf


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
train_dir='../train'#train verisi için dosya yolu
test_dir='../test'#test verisi için dosya yolu


def open_dir(data_dir):
    if not os.path.exists(data_dir):#böyle bir dosya yolunda dosya yoksa 
        os.mkdir(data_dir)#böyle bir dosya yolu olan dosya oluşturuyor

        
open_dir(train_dir)#open_dir fonksiyonu çağrıldı
open_dir(test_dir)
    
list_num=[4,11,33,52,77,82]#cifar100 datasından alinacak class numaraları
list_name=['/beaver','/boy','/forest','/oak_tree','/snail','/sunflower']#class isimleri

def subsetdata(X_data,y_data,image_dir):
    for i,num in enumerate(list_num):
        if not os.path.exists(image_dir+list_name[i]):
            index=np.where(y_data==num)
            subset_x_data=X_data[np.isin(y_data,[num]).flatten()]
            for a,x in enumerate(subset_x_data):
                image_path=(image_dir+list_name[i])
                open_dir(image_path)
                image_path=(image_path+"/"+str(a)+".png")
                cv2.imwrite(image_path,x)
        else:
            print(image_dir+list_name[i]+" konumunda dosya vardır kontrol ediniz\n")

subsetdata(X_train,y_train,train_dir)
subsetdata(X_test,y_test,test_dir)