import tensorflow as tf
import os

class TrainWrapper():
    def __init__(self, model, conf, train_dataset, val_dataset, test_dataset):
        self.net = model
        self.conf = conf
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.test_ds = test_dataset

    def compile_model(self):
        self.net.compile(
            optimizer=self.conf['optimizer'],
            loss=self.conf['loss'],
            metrics=self.conf['metrics']
            )

    def train(self, weights_path='./checkpoint/checkpoint.tf'):
        if not os.path.isdir(os.path.dirname(weights_path)):
            os.makedirs(os.path.dirname(weights_path))

        # define the callback
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor=self.conf['monitor_metric']
        )

        self.compile_model()

        self.net.fit(
            self.train_ds,
            batch_size = self.conf['batch_size'],
            epochs = self.conf['epochs'],
            validation_data = self.val_ds,
            callbacks = [checkpoint_cb]
        )

    def evaluate(self, weights_path = None):
        if weights_path is not None:
            self.net.load_weights(weights_path)

        self.net.evaluate(
            self.test_ds
        )
