import base64
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import adam
from keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf

import cv2
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import os
import pathlib
import numpy as np

triggers_dir=r"triggers.txt"
ingredients_dir=r"ingredients_simplified.txt"
#directory will be determined based on the output of the food classifier model
directory=r"F:\UPMC_Food101.tar\texts_txt\baby_back_ribs"
dir=pathlib.Path(directory)
file_base=dir.parts[3]
model_dir=r"Foodclassifier-016-5_08.h5"
img_dir=r"127721.jpg"
img_dim=224
#reading the triggers from file 
triggers=[]
with open(triggers_dir,encoding="utf8") as trgs:
       for line in trgs:
           for word in line.split(","):
               triggers.append(word)
print(triggers)
#getting the ingredients from imgredients list
all_ingredients=[]
with open(ingredients_dir,encoding="utf8") as ings:
    for line in ings:
        dish=[]
        for word in line.rstrip("\n").split(","):
            dish.append(word)
        all_ingredients.append(dish)
print(all_ingredients)


def ag_counter(idx,trigger_list):
    dish=all_ingredients[idx]
    c=len(set(dish).intersection(trigger_list))
    return c 

def get_model():
# create the base pre-trained model
    global model
    base_model = Xception(weights=None, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(2, activation='softmax')(x)
    
    # this is the model we will train
    Adam = adam(lr=0.0001, decay=1e-6)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam,metrics=['accuracy'])
    model.load_weights(model_dir)
    global graph
    graph = tf.get_default_graph()
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image

print(" * Loading Keras model...")
get_model()
img=cv2.imread(img_dir)
img = cv2.resize(img,(img_width,img_height))
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

for i in range(0,101):
    c=ag_counter(i, triggers)
    if(c>0):
        print(c," Allergen ingredients are present in this food.")
    else:
        print("Allergen ingredients are not present in this food.")

