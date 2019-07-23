import warnings
warnings.filterwarnings("ignore")
import base64
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
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
from werkzeug.utils import secure_filename
import os
import pathlib
import numpy as np
import easygui as e

app = Flask(__name__)
upload_dir=r""
app.config['UPLOAD_FOLDER']=upload_dir
triggers_dir=r"triggers.txt"
ingredients_dir=r"ingredients_simplified.txt"
classes_dir=r"labels.txt"

model_dir=r"fcx2107_FCXception-005-21_07.h5"

img_dim=224
#reading the triggers from file 
triggers=[]
with open(triggers_dir,encoding="utf8") as trgs:
       for line in trgs:
           for word in line.split(","):
               triggers.append(word)

#getting the ingredients from imgredients list
all_ingredients=[]
with open(ingredients_dir,encoding="utf8") as ings:
    for line in ings:
        dish=[]
        for word in line.rstrip("\n").split(","):
            dish.append(word)
        all_ingredients.append(dish)

labels=[]
with open(classes_dir,encoding="utf8") as foods:
       for line in foods:
           labels.append(line.rstrip("\n"))


def ag_counter(idx):
    dish=all_ingredients[idx]
    lst=list(set(dish).intersection(triggers))
    c=len(set(dish).intersection(triggers))
    return lst,c 
def ag_counter_topn(idxs):
    alg=[]
    for i,idx in enumerate(idxs):
        lst,_=ag_counter(idx)
        alg.extend(lst)
    return list(set(alg)),len(alg)

def get_model():
# create the base pre-trained model
    global model
    base_model = Xception(weights=None, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(101, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(model_dir)
    global graph
    graph = tf.get_default_graph()
    print(" * Model loaded!")

def preprocess_image(img_dir):
    img=cv2.imread(img_dir)
    img = cv2.resize(img,(img_dim,img_dim))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img=cv2.normalize(img.astype('float'),img,0,1,cv2.NORM_MINMAX)
    img = np.reshape(img,[1,224,224,3])
    return img
def get_top_n_acc(pred,n):
    return np.argpartition(pred[0],-n)[-n:][::-1]#returns index of top3 elements in descending order,for assending leave the last list silcing part


print(" * Loading Keras model...")
get_model()
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return 'No selected file'
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_dir=os.path.join(upload_dir,filename)
            img=preprocess_image(img_dir)
            with graph.as_default():
                pred=model.predict(img)
                
            top1=pred.argmax()
            print(top1,labels[top1])
            top3=get_top_n_acc(pred, 3)
            
            ing_list,c=ag_counter(top1)
            lst,d=ag_counter_topn(top3)
            b=labels[top1]
            print(*lst,d)
            a=' '.join(lst)
            
            if(c>0):
                response = {
                    'prediction':{  
                  'the dish is': b,
                  'Allergen warning ': a          
                }
            }
            else:
                response = {
                    'prediction' :
                {
                                    'This dish is safe to eat.'
                }}
            return jsonify(response)
            
if __name__=='__main__':
    app.run()            