import os,cv2
import pathlib
from PIL import Image as im
import numpy as np
image_dim=224

path=r"F:\images\\"
path=pathlib.Path(path)
for img_file in path.rglob("*.jpg"):
   img = im.open(img_file)
   if(np.shape(img)==(224,224,3)):
       pass
   else:
       print(np.shape(img))
       img=img.resize((image_dim,image_dim),im.ANTIALIAS)
       img=img.convert("RGB")
       img.save(img_file)
       