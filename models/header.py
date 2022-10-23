import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPool2D, Activation

class HeadWrapper(Layer):
    def __init__(self, last_dim, **kwargs):
        super().__init__(**kwargs)
        self.last_dim = last_dim
    
    def get_config(self):
        config = super().get_config()
        config.update({"last_dim": self.last_dim})
        return config
    
    def call(self, inputs):
        last_dim = self.last_dim
        batch_size = tf.shape(inputs[0])[0]
        outputs = []
        for conv_layer in inputs:
            outputs.append(tf.reshape(conv_layer, (batch_size, -1, last_dim)))
        
        return tf.concat(outputs, axis=1)

def get_head_from_outputs(hyper_params, outputs):
    total_labels = hyper_params["total_labels"]
    len_aspect_ratios =[len(x) + 1 for x in hyper_params["aspect_ratios"]]
    labels_head = []
    boxes_head = []
    for i, output in enumerate(outputs):
        aspect_ratio = len_aspect_ratios[i]
        labels_head.append(Conv2D(aspect_ratio * total_labels, (3,3), padding="same", name="{}_conv_label_output".format(i+1))(output))
        boxes_head.append(Conv2D(aspect_ratio * 4, (3, 3), padding="same", name="{}_conv_boxes_output".format(i+1))(output))

    pred_labels = HeadWrapper(total_labels, name="labels_head")(labels_head)
    pred_labels = Activation("softmax", name="conf")(pred_labels)

    pred_deltas = HeadWrapper(4, name="loc")(boxes_head)
    return pred_deltas, pred_labels