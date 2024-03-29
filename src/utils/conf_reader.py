import json
import tensorflow as tf
import albumentations as A
from augmentation_pipeline import get_transformation_pipeline
from losses import LOSSES

OPTIMIZERS = {
    'adam':tf.keras.optimizers.Adam,
    'sgd':tf.keras.optimizers.SGD
}

METRICS = {
    'mIoU':tf.keras.metrics.MeanIoU(6),
    'oneHotIoU':tf.keras.metrics.OneHotIoU(6, range(6)),
    'oneHotMeanIoU':tf.keras.metrics.OneHotMeanIoU(6),
    'mse':tf.keras.metrics.MeanSquaredError()
}

AUGMENTATION = {
    "toGray":A.ToGray(),
    "horizontalFlip":A.HorizontalFlip(),
    "verticalFlip":A.VerticalFlip(),
    "clahe":A.CLAHE()
}

def read_conf(filename):
    with open(filename,'r') as f:
        conf = json.load(f)
    
    # convert conf
    if not 'lr' in conf:
        conf['lr'] = 1e-3
    else:
        conf['lr'] = float(conf['lr'])

    if not 'optimizer' in conf:
        conf['optimizer'] = OPTIMIZERS['adam'](learning_rate=conf['lr'])   
    else:
        conf['optimizer'] = OPTIMIZERS[conf['optimizer']](learning_rate=conf['lr'])

    if not 'batch_size' in conf:
        conf['batch_size'] = 1
    else:
        conf['batch_szie'] = int(conf['batch_size'])

    if not 'metrics' in conf:
        conf['metrics'] = [METRICS['mIoU']]
    else:
        conf['metrics'] = [METRICS[x] for x in conf['metrics']]

    if not 'epochs' in conf:
        conf['epochs'] = 100
    else:
        conf['epochs'] = int(conf['epochs'])

    if not 'loss' in conf:
        conf['loss'] = LOSSES['categorical_cross_entropy']()
    else:
        conf['loss'] = LOSSES[conf['loss']]()

    if not 'augmentation' in conf:
        conf['augmentation'] = A.Compose([])
    else:
        conf['augmentation'] = [(AUGMENTATION[x], p) for x, p in conf['augmentation'].items()]
        conf['augmentation'] = get_transformation_pipeline(conf['augmentation'])

    return conf