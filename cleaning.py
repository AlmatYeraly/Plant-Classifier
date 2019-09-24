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

test_imgs = ['../test/{}'.format(i) for i in os.listdir(test_dir)]

train_imgs = train_aloe + train_peace
random.shuffle(train_imgs)

del train_aloe
del train_peace
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
            y.append(1)
        elif 'peace' in image:
            y.append(0)

    return X, y

X, y = read_and_process_image(train_imgs)

X[0]
y[0]

import seaborn as sns
del train_imgs
gc.collect()

X = np.array(X)
y = np.array(y)

sns.countplot(y)
plt.title('Labels for aloevera and peacelily')

print("shape of train images is: ", X.shape)
print("shape of labels is: ", y.shape)


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state=2)

print("shape of train images is: ", X_train.shape)
print("shape of validation images is: ", X_val.shape)
print("shape of labels (train) is: ", y_train.shape)
print("shape of labels (val) is: ", y_val.shape)

del X
del y
gc.collect()

ntrain = len(X_train)
nval = len(X_val)
batch_size = 16

