from telnetlib import GA
import tensorflow as tf
from glob import glob
from utils.parser import parse_arguments
from datasets.dataset import AMDdataset
from network.unet import UNet
from network.gan import GAN
from utils.train_utils import TrainWrapper
from utils.conf_reader import read_conf
import numpy as np


MODELS={
    'unet':UNet,
    'gan':GAN
}

if __name__ == '__main__':
    # make deterministic
    np.random.seed(0)
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    # parsing the arguments
    args = parse_arguments()
    # read config file
    conf = read_conf(args.conf)
    # opening the dataset
    datasets = AMDdataset(args.dataset_folder)
    datasets.build()
    # define the model
    net = MODELS[args.model]()
    # train
    train_wrapper = TrainWrapper(net, conf=conf, train_dataset=datasets.train, val_dataset=datasets.val, test_dataset=datasets.test)

    train_wrapper.train(args.weights_path)

    # evaluate
    train_wrapper.evaluate(args.weigths_path)
