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




train_dir='../train'
list_name=['/beaver','/boy','/forest','/oak_tree','/snail','/sunflower']

def image_show(train_dir,list_name):
     c=1
     for x,name in enumerate(list_name):
         img_list=[]
         img_first=name+"/0.png"
         img_concate=cv2.imread(train_dir+img_first)
         a=x*10
         
         for i in range(10):
             img_name=name+"/"+str(i)+".png"
             img=cv2.imread(train_dir+img_name)
             ax2 =plt.subplot(6,10,i+1+a)
             if (i+1+a)==c:
                 ax2.set_ylabel(name[1:],rotation=0,labelpad=25)
                 ax2.set_yticks([])
                 ax2.set_xticks([])
                 c=c+10
             else:
                 ax2.set_yticks([])
                 ax2.set_xticks([])

             plt.imshow(img)
             
    


image_show(train_dir,list_name) 
