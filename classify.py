import argparse
parser = argparse.ArgumentParser(description="Copies images into a yes and no subdirectory based on the model classification.")
parser.add_argument('model', type=str,
                    help="Filename of the pretrained model.")
parser.add_argument('data', type=str,
                    help="Directory with images.")
args = vars(parser.parse_args())

### settings

model_filename = args['model']
data_dir = args['data']

### imports
import tensorflow as tf
from os import scandir
from os import mkdir
from os.path import isdir
from shutil import copyfile
from tensorflow.keras import preprocessing as pp
import numpy as np
# colored terminal messages with supported
try:
    from termcolor import colored
except:
    def colored(a,b):
        return a 

### do the directories exist?
if not isdir(data_dir):
    print("Directory",data_dir,"does not exist / is not a directory.")
    exit(1)
yesdir = f"{data_dir}/yes"
if not isdir(yesdir):
    print("Making",yesdir)
    mkdir(yesdir)
nodir = f"{data_dir}/no"
if not isdir(nodir):
    print("Making",nodir)
    mkdir(nodir)

print('Loading model...')
model = tf.keras.models.load_model(model_filename)
model.summary()

### loop over all files)
for file in scandir(data_dir):
    if file.is_file():
        #f = f"{data_dir}/{path.path}"
        if not file.name.endswith('.jpg'): continue
        img = pp.image.load_img(file.path)
        x = pp.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)[0][0]
        if pred >= 0.5:
            target =  f"{data_dir}/yes/{file.name}"
            print(f"Score {pred:.3f} - copying to", colored(target,'green'))
        else:
            target =  f"{data_dir}/no/{file.name}"
            print(f"Score {pred:.3f} - copying to", colored(target,'red'))
        copyfile(file.path, target)

print("done :)")