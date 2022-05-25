import tensorflow as tf
from glob import glob
import os

REQUIRED_FOLDERS = ['Defects', 'NoDefects', 'Annotations']

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
