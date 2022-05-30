import json
import tensorflow as tf
from glob import glob
import os
import numpy as np
from PIL import Image, ImageDraw
from utils.visualize import apply_mask_on_image

REQUIRED_FOLDERS = ['Defects', 'NoDefects', 'annotations']

LABEL_DICT = {
    'BG':0,
    'HOLE':1,
    'VERTICAL':2,
    'HORIZONTAL':3,
    'SPATTERING':4,
    'INCANDESCENCE':5
}

all_defects = []

def map_fn(file:str, save_defects=False):
    # load image
    img = Image.open(file)
    shape = img.size
    # img = img.resize((256,256), 0)
    # build labels
    label = np.zeros((img.size[1], img.size[0]))
    if 'NoDefects' not in file:
        label = np.zeros((shape[1], shape[0]))
        # load file
        annotation_file = file.replace('Defects', 'annotations').replace('jpg', 'json')
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        # build annotation image & draw poligons
        label_img = Image.fromarray(label)
        for shape in annotations['shapes']:
            points = [(x[0], x[1]) for x in shape['points']]
            label_id = LABEL_DICT[shape['label'].upper()]
            # append defects to all defects
            if save_defects:
                all_defects.append((label_id, points))
            # draw poligon on image
            ImageDraw.Draw(label_img).polygon(points, fill=label_id)
        # label = np.array(label_img.resize((256,256), 0), dtype=np.uint8)
        label = np.array(label_img, dtype=np.uint8)
    label = tf.one_hot(label, len(LABEL_DICT))
    return (img, label)
    
def make_dataset(tuples):
    x = [tf.cast(tf.convert_to_tensor(np.array(t[0]).reshape((t[0].size[1],t[0].size[0], 1))), tf.float32) / 255.0 for t in tuples]
    y = [t[1] for t in tuples]

    return tf.data.Dataset.from_tensor_slices((x, y))

class AMDdataset():
    """Additive Manufactoring dataset class"""

    def __init__(self, path, image_shape=(1280, 1024, 3)):
        self.path = path
        self.image_shape = image_shape

    def build(self):
        # load images and annotations
        folders = [x for x in os.listdir(self.path) if os.path.basename(x) in REQUIRED_FOLDERS]

        if len(folders) != len(REQUIRED_FOLDERS):
            raise FileNotFoundError(f'Directory {self.path} does not contain correct folders. It must contains {REQUIRED_FOLDERS}')
        
        train = []
        test = []
        val = []

        # split folders
        for f in folders:
            files = np.array(glob(os.path.join(self.path, f,'*.jpg')))
            if len(files) == 0:
                continue
            n = len(files) // 9
            idx = np.random.permutation(np.arange(len(files)))

            test.extend(files[idx[:3*n]]) # 3/9
            val.extend(files[idx[3*n:5*n]]) # 2/9
            train.extend(files[idx[5*n:]])  # 4/9

        train = [map_fn(x, True) for x in train]
        test = [map_fn(x) for x in test] 
        val = [map_fn(x, True) for x in val] 

        # build the datasets
        self.train = make_dataset(train)
        self.test = make_dataset(test)
        self.val = make_dataset(val)