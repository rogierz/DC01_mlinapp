import json
import tensorflow as tf
from utils.losses import LOSSES

OPTIMIZERS = {
    'adam':tf.keras.optimizers.Adam,
    'sgd':tf.keras.optimizers.SGD
}

METRICS = {
    'mIoU':tf.keras.metrics.MeanIoU(6)
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
        conf['loss'] = LOSSES['binary_crossentropy']()
    else:
        conf['loss'] = LOSSES[conf['loss']]()

    return conf