import tensorflow as tf

class ConvBlock(tf.keras.layers.Layer):
    KERNEL_SIZE = 3
    def __init__(self, filters=64, name="conv_block", with_upsample=False, with_downsample=False, **kwargs):
        super(ConvBlock, self).__init__(name=name, **kwargs)
        self.with_upsample = with_upsample
        self.with_downsample = with_downsample

        if with_upsample:
            self.upsample = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding="same")

        if with_downsample:
            self.downsample = tf.keras.layers.MaxPool2D()

        self.conv_1 = tf.keras.layers.Conv2D(filters, self.KERNEL_SIZE, padding="same")
        self.batch_1 = tf.keras.layers.BatchNormalization()
        self.conv_2 = tf.keras.layers.Conv2D(filters, self.KERNEL_SIZE, padding="same")
        self.batch_2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()


    def call(self, inputs):
        other_inputs = None

        if type(inputs) is tuple:
            inputs, other_inputs = inputs

        if self.with_downsample:
            inputs = self.downsample(inputs)

        if other_inputs is not None:
            if self.with_upsample:
                inputs = self.upsample(inputs)
            inputs = tf.concat([inputs, other_inputs], axis=-1)
        
        x = self.conv_1(inputs)
        x = self.batch_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu(x)
        return x

class UNet(tf.keras.Model):
    def __init__(
        self,
        name="unet",
        num_classes=6,
        input_shape=(256, 256),
        **kwargs
    ):
        super(UNet, self).__init__(name=name,**kwargs)
        self.num_classes = num_classes
        self.encode_1 = ConvBlock(filters=64)
        self.encode_2 = ConvBlock(filters=128, with_downsample=True)
        self.encode_3 = ConvBlock(filters=256, with_downsample=True)
        self.encode_4 = ConvBlock(filters=512, with_downsample=True)

        self.latent = ConvBlock(filters=1024, with_downsample=True)

        self.decode_4 = ConvBlock(filters=512, with_upsample=True)
        self.decode_3 = ConvBlock(filters=256, with_upsample=True)
        self.decode_2 = ConvBlock(filters=128, with_upsample=True)
        self.decode_1 = ConvBlock(filters=64, with_upsample=True)
        self.result = tf.keras.layers.Conv2D(self.num_classes, 1, activation='sigmoid', padding="same")

    def call(self, inputs):
        x1 = self.encode_1(inputs)
        x2 = self.encode_2(x1)
        x3 = self.encode_3(x2)
        x4 = self.encode_4(x3)
        x = self.latent(x4)
        x = self.decode_4((x, x4))
        x = self.decode_3((x, x3))
        x = self.decode_2((x, x2))
        x = self.decode_1((x, x1))
        x = self.result(x)
        return x