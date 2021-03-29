import tensorflow as tf
from keras.layers.advanced_activations import PReLU

HEIGHT = 384
WIDTH = 768
CLASS_N = 24


class channelAttention(tf.keras.Model):
    """
    https://github.com/lRomul/argus-freesound/blob/master/src/models/simple_attention.py

    https://paperswithcode.com/method/channel-attention-module#

    avg pooling input as well as separate max pooling input which connects to a common
    FC layer (instead of using MLP, Conv2D layer is used), sum them up and add a sigmoid
    activation 
    """

    def __init__(self, in_planes, ratio=16):
        """
        Parameters
        ----------

        in_planes: input channel size. We need to define this because the attention layers
        are after the convblocks which we should know the output channel size based on our
        filter size 
        """
        super().__init__()
        self.in_channel = in_planes

        # this is equals to global pooling
        # AdaptiveAvgPool2d(1)
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPool2D()

        self.fc1 = tf.keras.layers.Conv2D(
            filters=in_planes//ratio, kernel_size=1, use_bias=False)
        self.relu1 = tf.keras.layers.ReLU()
        self.fc2 = tf.keras.layers.Conv2D(
            filters=in_planes, kernel_size=1, use_bias=False)

    def call(self, inputs):
        # we reshape after pooling to emulate the pytorch adaptive pooling (1) which results in (channel,1,1)
        # for tensorflow, we want it to be (None,1,1,24) where 24 is the channel number
        avg_out = tf.reshape(self.avg_pool(
            inputs), (-1, 1, 1, self.in_channel))
        avg_out = self.fc2(self.relu1(self.fc1(avg_out)))
        max_out = tf.reshape(self.max_pool(
            inputs), (-1, 1, 1, self.in_channel))
        max_out = self.fc2(self.relu1(self.fc1(max_out)))
        out = avg_out + max_out
        return tf.keras.activations.sigmoid(out)


class spatialAttention(tf.keras.Model):

    def __init__(self, kernel_size=7):
        super().__init__()

        # some checks in the repo
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # keras does not allow us to define the padding in conv2d, instead we do padding, the
        self.padding = tf.keras.layers.ZeroPadding2D(padding=padding)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=1, kernel_size=kernel_size, use_bias=False)

    def call(self, inputs):
        # here, we squeeze the time axis rather than channel
        # resulting attention multiply will be apply the time axis
        # i.e we are searching for the "best" y axis (frequency axis)
        avg_out = tf.reduce_mean(inputs, axis=1, keepdims=True)
        max_out = tf.reduce_max(inputs, axis=1, keepdims=True)

        out = tf.keras.layers.concatenate([avg_out, max_out])
        out = self.conv1(self.padding(out))

        return tf.keras.activations.sigmoid(out)


class ConvolutionalBlockAttentionModule(tf.keras.Model):
    """
    in_planes: expected shape of the last dim (usually channel)
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()

        self.ca = channelAttention(in_planes=in_planes, ratio=ratio)
        self.sa = spatialAttention(kernel_size=kernel_size)

    def call(self, inputs):
        # multiple the input with a sigmoid score
        # (in attention models, it is the softmax score)
        out = self.ca(inputs) * inputs
        out = self.sa(inputs) * out
        return out


class ConvBlock(tf.keras.Model):

    def __init__(self, out_channels):
        super().__init__()

        self.padding = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=3, strides=1, use_bias=False)
        self.batch_normalization_1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.padding2 = tf.keras.layers.ZeroPadding2D(padding=1)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=out_channels, kernel_size=3, strides=1, use_bias=False)
        self.batch_normalization_2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.avg_pool = tf.keras.layers.AveragePooling2D()

    def call(self, inputs):
        x = self.padding(inputs)
        x = self.conv1(x)
        x = self.batch_normalization_1(x)
        x = self.relu1(x)

        x = self.padding2(x)
        x = self.conv2(x)
        x = self.batch_normalization_2(x)
        x = self.relu2(x)
        # x = tf.nn.avg_pool2d(x,2,2,"VALID") # follows pytorch default
        x = self.avg_pool(x)

        return x


class my_model(tf.keras.Model):

    def __init__(self, num_class=24, base_size=64, ratio=16, kernel_size=7, drop_out=0.4):
        super().__init__()

        # define our pre-trained model
        # we dont want the head and want to define our FCL
        # here, simply change the model as you want
        self.base_model = tf.keras.applications.DenseNet201(
            include_top=False, weights="imagenet", input_shape=(HEIGHT, WIDTH, 3))

        # we want to train only the FCL
        # if our config defined None, we will freeze all, else only freeze the first [:layers]
        if cfg['model_params']['freeze_to'] is None:
            for layer in self.base_model.layers:
                layer.trainable = False
        else:
            # cfg defaults to :0
            # means we want to freeze nothing
            for layer in self.base_model.layers[:0]:
                layer.trainable = False
        self.conv1 = ConvBlock(out_channels=base_size*4)
        self.conv = ConvBlock(out_channels=base_size*8)
        self.avg_pool = tf.keras.layers.AveragePooling2D()
        self.attention = self.attention = ConvolutionalBlockAttentionModule(
            in_planes=512, ratio=ratio, kernel_size=kernel_size)  # last dim from convo block is 512
        self.global_avg_pooling2d = tf.keras.layers.GlobalAveragePooling2D()
        self.batch_normalization_0 = tf.keras.layers.BatchNormalization()
        self.batch_normalization_1 = tf.keras.layers.BatchNormalization()
        self.batch_normalization_2 = tf.keras.layers.BatchNormalization()
        self.drop_out_1 = tf.keras.layers.Dropout(drop_out)
        self.drop_out_2 = tf.keras.layers.Dropout(drop_out)

        self.dense1 = tf.keras.layers.Dense(base_size * 2, activation=PReLU())
        self.dense2 = tf.keras.layers.Dense(CLASS_N)

    def call(self, inputs):
        y = self.base_model(inputs['input_1'])
        y = self.batch_normalization_0(y)
        y = self.conv1(y)
        y = self.conv(y)
        y = self.attention(y)
        y = self.global_avg_pooling2d(y)
#         y = self.batch_normalization_1(y)
        y = self.drop_out_1(y)
        y = self.dense1(y)
        y = self.batch_normalization_2(y)
        y = self.drop_out_2(y)
        y = self.dense2(y)
        return y
