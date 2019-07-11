from shutil import copytree as copydir
import os

src=r"F:\images\\"
dst=r"F:\Food_imgs_dataset\\"


copydir(src,dst,symlinks=True)
if (os.path.isdir(dst)):
    pass
else:
    pass
    os.makedirs(dst, mode=0o777, exist_ok=True)
    copydir(src,dst,symlinks=True)