import tensorflow as tf
from glob import glob
from tensorflow.keras.layers import Layer, Conv2D, MaxPool2D, Input
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from .header import get_head_from_outputs

class L2Normalization(Layer):
    def __init__(self, scale_factor, **kwargs):
        super().__init__(**kwargs)
        self.scale_factor = scale_factor
    
    def get_config(self):
        config = super().get_config()
        config.update({"scale_factor": self.scale_factor})
        return config

    def build(self, input_shape):
        init_scale_factor = tf.fill((input_shape[-1],), float(self.scale_factor))
        self.scale = tf.Variable(init_scale_factor, trainable=True)

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=-1) * self.scale

def get_model(hyper_params):
    scale_factor = 20.0
    reg_factor = 5e-4
    initializer = "glorot_normal"
    activation = "relu"
    regularizer=l2(reg_factor)

    input = Input(shape=(None, None, 3), name="input")
    
    # vgg16 layers
    conv1_1 = Conv2D(64, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv1_1")(input)
    conv1_2 = Conv2D(64, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv1_2")(conv1_1)
    pool1 = MaxPool2D((2,2), strides=(2,2), padding="same", name="pool1")(conv1_2)

    conv2_1 = Conv2D(128, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv2_1")(pool1)
    conv2_2 = Conv2D(128, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv2_2")(conv2_1)
    pool2 = MaxPool2D((2,2), strides=(2,2), padding="same", name="pool2")(conv2_2)

    conv3_1 = Conv2D(256, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv3_1")(pool2)
    conv3_2 = Conv2D(256, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv3_2")(conv3_1)
    conv3_3 = Conv2D(256, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv3_3")(conv3_2)
    pool3 = MaxPool2D((2,2), strides=(2,2), padding="same", name="pool3")(conv3_3)

    conv4_1 = Conv2D(512, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv4_1")(pool3)
    conv4_2 = Conv2D(512, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv4_2")(conv4_1)
    conv4_3 = Conv2D(512, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv4_3")(conv4_2)
    pool4 = MaxPool2D((2,2), strides=(2,2), padding="same", name="pool4")(conv4_3)

    conv5_1 = Conv2D(512, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv5_1")(pool4)
    conv5_2 = Conv2D(512, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv5_2")(conv5_1)
    conv5_3 = Conv2D(512, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv5_3")(conv5_2)
    pool5 = MaxPool2D((3,3), strides=(1,1), padding="same", name="pool5")(conv5_3)

    # convert fc6, fc7 to conv6, conv7 and remove dropout
    conv6 = Conv2D(1024, (3,3), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv6")(pool5)
    # conv6 = Conv2D(1024, (3,3), dilation_rate=6, padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv6")(pool5)
    conv7 = Conv2D(1024, (1,1), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv7")(conv6)

    # extra layers
    conv8_1 = Conv2D(256, (1,1), padding="valid", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv8_1")(conv7)
    conv8_2 = Conv2D(512, (3,3), strides=(2,2), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv8_2")(conv8_1)
    
    conv9_1 = Conv2D(128, (1,1), padding="valid", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv9_1")(conv8_2)
    conv9_2 = Conv2D(256, (3,3), strides=(2,2), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv9_2")(conv9_1)

    conv10_1 = Conv2D(128, (1,1), padding="valid", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv10_1")(conv9_2)
    conv10_2 = Conv2D(256, (3,3), padding="valid", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv10_2")(conv10_1)

    conv11_1 = Conv2D(128, (1,1), padding="same", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv11_1")(conv10_2)
    conv11_2 = Conv2D(256, (3,3), padding="valid", activation=activation, kernel_initializer=initializer, kernel_regularizer=regularizer, name="conv11_2")(conv11_1)

    conv4_3_norm = L2Normalization(scale_factor)(conv4_3)

    pred_deltas, pred_labels = get_head_from_outputs(hyper_params, [conv4_3_norm, conv7, conv8_2, conv9_2, conv10_2, conv11_2])
    return Model(inputs=input, outputs=[pred_deltas, pred_labels])

def init_model(model):
    model(tf.random.uniform((1, 300, 300, 3)))