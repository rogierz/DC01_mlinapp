import json
import tensorflow as tf
from glob import glob
import os
import numpy as np
from PIL import Image, ImageDraw

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
    img = tf.keras.utils.load_img(file)
    shape = img.size
    # build labels
    label = np.zeros((shape[0], shape[1]))
    if 'NoDefects' not in file:
        # load file
        annotation_file = file.replace('Defects', 'annotations').replace('jpg', 'json')
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        # build annotation image & draw poligons
        img = Image.fromarray(label)
        for shape in annotations['shapes']:
            # use only one label name
            if shape['label'].upper() == 'VERTICAL DEFECT':
                shape['label'] = 'VERTICAL'                
            if shape['label'].upper() == 'SPATTING':
                shape['label'] = 'SPATTERING'

            points = [(x[0], x[1]) for x in shape['points']]
            label_id = LABEL_DICT[shape['label'].upper()]
            # append defects to all defects
            if save_defects:
                all_defects.append((label_id, points))
            # draw poligon on image
            ImageDraw.Draw(img).polygon(points, fill=label_id)
        label = np.array(img)
    label = tf.one_hot(label, len(LABEL_DICT))
    return (img, label)
    

class AMDdataset():
    '''Additive Manufactoring dataset class'''

    def __init__(self, path, image_shape=(1280, 1024, 3)):
        self.path = path
        self.image_shape = image_shape

    def build(self):
        # load images and annotations
        folders = [x for x in os.listdir(self.path) if os.path.basename(x) in REQUIRED_FOLDERS]

        if len(folders) != len(REQUIRED_FOLDERS):
            raise FileNotFoundError(f'Directory {self.path} does not contain correct folders. It must contains {REQUIRED_FOLDERS}')
        
        # TODO split the dataset and load it into 3 tf.dataset: self.train, self.val, self.test
        train = []
        test = []
        val = []

        # split folders
        for f in folders:
            files = np.array(glob(os.path.join(self.path, f,'*.jpg')))
            if len(files) == 0:
                continue
            n = len(files) // 3
            idx = np.random.permutation(np.arange(len(files)))

            test.extend(files[idx[:n]])
            val.extend(files[idx[n:n*2]])
            train.extend(files[idx[n*2:]])

        train = [map_fn(x, True) for x in train]
        test = [map_fn(x) for x in test]
        val = [map_fn(x, True) for x in val]

