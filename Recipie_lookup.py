
import os
import pathlib
import numpy as np

triggers=['Milk', 'Egg', 'Peanut', 'Tree nut', 'Soy', 'Wheat','flour' ,'Fish', 'Shellfish', 'Legumes']
directory=r"F:\UPMC_Food101.tar\texts_txt\baby_back_ribs"

dir=pathlib.Path(directory)
file_base=dir.parts[3]

def make_uri(filenumber):
    return directory+"\\"+file_base+"_"+filenumber+".txt"
def foodcounter(filename, trigger_list):
    with open(filename,encoding="utf8") as file_object:
        file_text = file_object.read()
        #Returns presence of ingridients in the recipe
        return [ 1 if file_text.count(word)>1 else file_text.count(word) for word in trigger_list] 
       # return [ file_text.count(word) for word in trigger_list] #counts number of occurence in the file


trigger_data=[]

for i,file in enumerate(os.listdir(directory)):
    trigger_data.append(foodcounter(os.path.join(directory,file), triggers))
    
   
print(np.sum(trigger_data[6],axis=0))
from openpyxl import Workbook
wb = Workbook()

# grab the active worksheet
ws = wb.active

# Data can be assigned directly to cells


# Rows can also be appended
ws.append(triggers)
for i,row in enumerate(trigger_data):
    ws.append(trigger_data[i])

# Save the file
wb.save("trigger_data_"+file_base+".xlsx")