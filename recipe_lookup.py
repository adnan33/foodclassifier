'''

'''
import tensorflow as tf
import os
import pathlib
import numpy as np

triggers=['milk', 'egg', 'peanut', 'nut', 'soy', 'wheat','flour' ,'fish', 'shellfish', 'legumes']
#directory will be determined based on the output of the food classifier model
directory=r"F:\UPMC_Food101.tar\texts_txt\baby_back_ribs"
dir=pathlib.Path(directory)
file_base=dir.parts[3]
#
def foodcounter(filename, trigger_list):
    with open(filename,encoding="utf8") as file_object:
        file_text = file_object.read()
        #Returns presence of ingridients in the recipe
        #return [ 1 if file_text.count(word)>1 else file_text.count(word) for word in trigger_list] 
       #counts number of occurence in the file
        return [ file_text.count(word) for word in trigger_list] 
print(foodcounter(r"F:\ingredients_simplified.txt", triggers))

for i,file in enumerate(os.listdir(directory)):
    data=foodcounter(os.path.join(directory,file), triggers)
    
    if(np.sum(data,axis=0)>0):
        print("Allergen ingredients are present in this food.")
    else:
        print("Allergen ingredients are not present in this food.")
       
# 
# restoring_graph = tf.Graph()
# with restoring_graph.as_default():
#     with tf.Session(graph=restoring_graph) as sess:
#        # Restore saved values
#        tf.saved_model.loader.load(
#           sess,
#           [tf.saved_model.tag_constants.SERVING],
#           r"F:\demo\Food_classifier\1"  # Path to SavedModel
#        )
