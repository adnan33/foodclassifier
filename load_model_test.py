
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import os,glob,fnmatch
from keras import applications
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.layers import Activation
from keras.optimizers import adam
from keras.metrics import categorical_crossentropy
from keras import optimizers
from sklearn.metrics import confusion_matrix
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D,AveragePooling2D
from keras import backend as K
plt.rcParams["axes.grid"] = False

img_width, img_height = 224,224
n_classes=101



K.clear_session()

base_model = InceptionV3(weights=None, include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = AveragePooling2D(pool_size=(8, 8))(x)
x = Dropout(.4)(x)
x = Flatten()(x)
predictions = Dense(101, activation="softmax", kernel_initializer="glorot_uniform")(x)

model = Model(inputs=base_model.input, outputs=predictions)
name=[]
for layer in model.layers:
    name.append(layer.name)
print(name)
model.load_weights(r"F:\model.h5")
