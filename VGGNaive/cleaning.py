### Almat Yeraly and Jude Battista
### This was directly copied and modified from https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9

import cv2
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt

import os
import random
import gc 

train_dir = './train'
test_dir = './test'

train_aloe = ['./train/{}'.format(i) for i in os.listdir(train_dir) if 'aloevera' in i]
train_peace = ['./train/{}'.format(i) for i in os.listdir(train_dir) if 'peacelily' in i]
train_spider = ['./train/{}'.format(i) for i in os.listdir(train_dir) if 'spider' in i]
train_cane = ['./train/{}'.format(i) for i in os.listdir(train_dir) if 'cane' in i]

test_imgs = ['./test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_aloe + train_peace + train_spider + train_cane
random.shuffle(train_imgs)

del train_aloe
del train_peace
del train_spider
del train_cane
gc.collect()

nrows = 150
ncolumns = 150
channels = 3


def read_and_process_image(list_of_images):
    X = []
    y = []

    for image in list_of_images:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), 
                (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))
        if 'aloe' in image:
            y.append(0)
        elif 'peace' in image:
            y.append(1)
        elif 'spider' in image:
            y.append(2)
        elif 'cane' in image:
            y.append(3)    

    return X, y

X, y = read_and_process_image(train_imgs)

import seaborn as sns
del train_imgs
gc.collect()

X = np.array(X)
y = np.array(y)

sns.countplot(y)
plt.title('Labels for aloevera, peacelily, spiderplant, dumbcane')

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state=2)

del X
del y
gc.collect()

ntrain = len(X_train)
nval = len(X_val)
batch_size = 16

