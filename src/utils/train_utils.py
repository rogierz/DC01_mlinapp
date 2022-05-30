import tensorflow as tf
from datetime import datetime

class TrainWrapper():
    def __init__(self, model, conf, train_dataset, val_dataset, test_dataset):
        self.net = model
        self.conf = conf
        self.train_ds = train_dataset.batch(conf['batch_size'])
        self.val_ds = val_dataset.batch(conf['batch_size'])
        self.test_ds = test_dataset.batch(conf['batch_size'])

    def compile_model(self):
        self.net.compile(
            optimizer=self.conf['optimizer'],
            loss=self.conf['loss'],
            metrics=self.conf['metrics']
            )

    def train(self, weights_path='./checkpoint/checkpoint.tf'):
        # if not os.path.isdir(os.path.dirname(weights_path)):
        #     os.makedirs(os.path.dirname(weights_path))

        # define the callback
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor=self.conf['monitor_metric'],
            save_best_only=True,
            save_weights_only=True
        )

        # early stopping callback 
        early_stop_cb = tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=1e-4,
            patience=10
        )

        # reduce lr on plateau callback
        reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss',
            factor=0.1,
            patience=5,
            min_delta=1e-3
        )

        # tensorboard callback
        log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, 
            histogram_freq=1
        )

        self.compile_model()

        self.net.fit(
            self.train_ds,
            batch_size = self.conf['batch_size'],
            epochs = self.conf['epochs'],
            validation_data = self.val_ds,
            callbacks = [checkpoint_cb, early_stop_cb, reduce_lr_cb, tensorboard_callback]
        )

    def evaluate(self, weights_path = None):
        if weights_path is not None:
            self.net.load_weights(weights_path)

        self.net.evaluate(
            self.test_ds
        )
