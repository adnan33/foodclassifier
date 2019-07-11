import os
import tensorflow as tf
import PIL.Image as image
import easygui
from matplotlib import pyplot as plt
import numpy as np
import csv
import keras.models as m
import keras
import pip
#print(keras.__version__)
#suppressing deprication warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

model_dir="F:\Food_model.h5"
labels_dir='F:\classes.csv'
ingredients_dir='F:\ingredients_simplified.csv'
triggers=['milk', 'egg', 'peanut', 'nut', 'soy', 'wheat','flour' ,'fish', 'shellfish', 'legumes']

#Loading keras model
model=m.load_model(model_dir)

#loading class labels and ingredients data
with open(ingredients_dir, 'rt',encoding="utf8") as f:
    reader = csv.reader(f,delimiter=",")
    ingredients= list(reader)
with open(labels_dir, 'rt',encoding="utf8") as f:
    reader = csv.reader(f,delimiter=",")
    classes= list(reader)




image_dim=299

#Giving image for the model to predict
img=image.open(easygui.fileopenbox())
img=img.resize((image_dim,image_dim),image.ANTIALIAS)
img=img.convert("RGB")
plt.axis('off')
plt.imshow(img)
plt.show()
img=np.array(img)
img=np.reshape(img, (1,299,299,3))
predictions=model.predict(img)

#getting the prediction to find the list of ingredients
food_class=np.argmax(predictions)
print("the dish in the image is {}".format(classes[food_class][0]))
items=list(ingredients[food_class])
a=list(items.count(word) for word in triggers)
op=dict(zip(triggers,a))
print(op)