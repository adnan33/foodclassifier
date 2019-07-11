import os,glob,pathlib
import shutil
from pathlib import WindowsPath
import numpy as np
src=r"F:\images"
dst=r"F:\dataset_v\\"
dest_train=dst+"train\\"
dest_valid=dst+"valid\\"
dest_test=dst+"test\\"
split_per=.1

def SubDirPath (d):
    return [f for f in d.iterdir() if f.is_dir()]

subdirs = SubDirPath(WindowsPath(src))
def copy_file(files,dest):
    for file in files:
        if os.path.isfile(file):
            if(os.path.isdir(dest)):
               shutil.copy2(file, dest)
            else:
                os.makedirs(dest,mode=0o777, exist_ok=True)
                shutil.copy2(file, dest)
labels_dict={}           
for i,dir in enumerate(subdirs):
    path=pathlib.Path(dir)
    subdir=path.parts[2]
    print(subdir)
    labels_dict.update({i:subdir})
print(labels_dict)
#     files = glob.iglob(os.path.join(dir, "*.jpg"))
#     files=list(files)
#     np.random.shuffle(files)
#     file_train_limit=int(len(files)*split_per)
#     file_valid_limit=2*file_train_limit
#     files_test=files[0:file_train_limit]
#     files_valid=files[file_train_limit:file_valid_limit]
#     files_train=files[file_valid_limit:]
#     
#     copy_file(files_train, dest_train+subdir+'\\')
#     copy_file(files_test, dest_test+subdir+'\\')
#     copy_file(files_valid, dest_valid+subdir+'\\')