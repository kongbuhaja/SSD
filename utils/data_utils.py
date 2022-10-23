import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import numpy as np
import xmltodict

def get_dataset(name, split, data_dir="~/tensorflow_datasets"):
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info

def get_custom_dataset(name, split, data_dir="C:/Users/HS/tensorflow_datasets"):
    assert split in ["train", "train+validation", "validation", "test"]
    dataset, info = tfds.load(name, split=split, data_dir=data_dir, with_info=True)
    return dataset, info

def preprocessing(image_data, final_height, final_width, augmentation_fn=None, evaluate=False):
    img = image_data["image"]
    gt_boxes = image_data["objects"]["bbox"]
    gt_labels = tf.cast(image_data["objects"]["label"] + 1, tf.int32)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (final_height, final_width))
    if evaluate:
        not_diff = tf.logical_not(image_data["objects"]["is_difficult"])
        gt_boxes = gt_boxes[not_diff]
        gt_labels = gt_labels[not_diff]
    if augmentation_fn:
        img, gt_boxes = augmentation_fn(img, gt_boxes)
    return img, gt_boxes, gt_labels

def get_total_item_size(info, split):
    assert split in ["train", "train+validation", "validation", "test"]
    if split == "train+validation":
        return info.splits["train"].num_examples + info.splits["validation"].num_examples
    return info.splits[split].num_examples

def get_labels(info):
    return info.features['objects']['label'].names

def custom_data_generator(img_paths, final_height, final_width):
    for img_path in img_paths:
        image = Image.open(img_path)
        resized_image = image.resize((final_width, final_height), Image.LANCZOS)
        img = np.array(resized_image)
        img = tf.image.convert_image_dtype(img, tf.float32)
        yield img, tf.constant([[]], dtype=tf.float32), tf.constant([], dtype=tf.int32)

def get_data_types():
    return tf.float32, tf.float32, tf.int32

def get_data_shapes():
    return ([None, None, None], [None, None], [None,])

def get_padding_values():
    return (tf.constant(0, tf.float32), tf.constant(0, tf.float32), tf.constant(-1, tf.int32))