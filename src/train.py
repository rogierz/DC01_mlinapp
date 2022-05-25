from telnetlib import GA
import tensorflow
from glob import glob
from utils.parser import parse_arguments
from datasets.dataset import AMDdataset
from network.unet import UNet
from network.gan import GAN
from utils.train_utils import TrainWrapper
from utils.conf_reader import read_conf

MODELS={
    'unet':UNet,
    'gan':GAN
}

if __name__ == '__main__':
    # parsing the arguments
    args = parse_arguments()
    # read config file
    conf = read_conf(args.conf)
    # opening the dataset
    datasets = AMDdataset(args.dataset_folder)
    # define the model
    net = MODELS(args.model)
    # train
    train_wrapper = TrainWrapper(net, conf=conf, train_dataset=datasets.train, val_dataset=datasets.val, test_dataset=datasets.test)

    train_wrapper.train(args.weights_path)

    # evaluate
    train_wrapper.evaluate(args.weigths_path)
