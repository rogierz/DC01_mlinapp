import argparse
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description="GAN for image harmonization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--dataset-folder", type=str, default="dataset", help="The folderwhere the dataset is contained. It must contain 3 folders: /Defects /NoDefects /annotations")
    parser.add_argument("--model", type=str, default="unet", help="the model to be compiled and trained.")
    parser.add_argument("--conf", type=str, default="./config/conf.json", help="the path to the json config file.")
    parser.add_argument("--weights-path", type=str, required=True, help="path where to save the model weights (if it does not exists will be created).")

    args = parser.parse_args()

    # check the received args
    if not os.path.isdir(args.dataset_folder):
        raise FileNotFoundError(f'The directory {args.dataset_folder} does not exists')
    if not os.path.isfile(args.conf):
        raise FileNotFoundError(f'The config file {args.conf} does not exist')
    if not os.path.isdir(os.path.dirname(args.weights_path)):
        os.makedirs(os.path.dirname(args.weights_path))
        
    return args
