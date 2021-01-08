#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 16:47:08 2020

@author: aycaburcu
"""


import json
import matplotlib.pyplot as plt
import cv2 
import numpy as np
import os
import tensorflow as tf




train_dir='/mnt/sdb2/ders/deep_learning_bsm/DeepLearning/Cifar100/train'
test_dir='/mnt/sdb2/ders/deep_learning_bsm/DeepLearning/Cifar100/test'
list_name=['/beaver','/boy','/forest','/oak_tree','/snail','/sunflower']

def image_show():
     concate_list=[]
     for x,name in enumerate(list_name):
         img_list=[]
         img_first=name+"/0.png"
         img_concate=cv2.imread(train_dir+img_first)
         for i in range(10):
             img_name=name+"/"+str(i)+".png"
             img=cv2.imread(train_dir+img_name)
             img_list.append(img)
             if i>0:
                 img_concate=np.concatenate((img_concate,img_list[i]),axis=1)
         concate_list.append(img_concate)
         ax3 =plt.subplot(6,1,x+1)
         ax3.set_yticks([])
         ax3.set_xticks([])
         ax3.set_ylabel(name[1:],rotation=0,labelpad=26)
         ax3.margins(x=0.25, y=-0.25)
         plt.imshow(concate_list[x])
         plt.axis('on')
         
image_show() 