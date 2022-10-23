import tensorflow as tf
from utils import bbox_utils

def apply(img, gt_boxes):
    color_methods = [random_brightness, random_contrast, random_hue, random_saturation]
    geometric_methods = [patch, flip_horizontally]
    for augmentation_method in geometric_methods + color_methods:
        img, gt_boxes = randomly_apply_operation(augmentation_method, img, gt_boxes)
    img = tf.clip_by_value(img, 0, 1)
    return img, gt_boxes

def get_random_bool():
    return tf.greater(tf.random.uniform((), dtype=tf.float32), 0.5)

def randomly_apply_operation(operation, img, gt_boxes, *args):
    return tf.cond(
        get_random_bool(),
        lambda: operation(img, gt_boxes, *args),
        lambda: (img, gt_boxes)
    )

def random_brightness(img, gt_boxes, max_delta=0.12):
    return tf.image.random_brightness(img, max_delta), gt_boxes

def random_contrast(img, gt_boxes, lower=0.5, upper=1.5):
    return tf.image.random_contrast(img, lower, upper), gt_boxes

def random_hue(img, gt_boxes, max_delta=0.08):
    return tf.image.random_hue(img, max_delta), gt_boxes

def random_saturation(img, gt_boxes, lower=0.5, upper=1.5):
    return tf.image.random_saturation(img, lower, upper), gt_boxes

def flip_horizontally(img, gt_boxes):
    flipped_img = tf.image.flip_left_right(img)
    flipped_gt_boxes = tf.stack([gt_boxes[...,0],
                                 1.0 - gt_boxes[..., 1],
                                 gt_boxes[..., 2],
                                 1.0 - gt_boxes[..., 3]], -1)
    return flipped_img, flipped_gt_boxes

def get_raondom_min_overlap():
    overlaps = tf.constant([0.1, 0.3, 0.5, 0.7, 0.9], dtype=tf.float32)
    i = tf.random.uniform((), minval=0, maxval=tf.shape(overlaps)[0], dtype=tf.int32)
    return overlaps[i]

def expand_image(img, gt_boxes, height, width):
    expansion_ratio = tf.random.uniform((), minval=1, maxval=4, dtype=tf.float32)
    final_height, final_width = tf.round(height * expansion_ratio), tf.round(width * expansion_ratio)
    pad_left = tf.round(tf.random.uniform((), minval=0, maxval=final_width-width, dtype=tf.float32))
    pad_top = tf.round(tf.random.uniform((), minval=0, maxval=final_height-height, dtype=tf.float32))
    pad_right = final_width - (width + pad_left)
    pad_bottom = final_height - (height + pad_top)

    mean, _ = tf.nn.moments(img, [0,1])
    expanded_image = tf.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right), (0,0)), constant_values=-1)
    expanded_image = tf.where(expanded_image == -1, mean, expanded_image)

    min_max = tf.stack([-pad_top, -pad_left, pad_bottom+height, pad_right+width], -1) / [height, width, height, width]
    modified_gt_boxes = bbox_utils.renormalize_bboxes_with_min_max(gt_boxes, min_max)

    return expanded_image, modified_gt_boxes

def patch(img, gt_boxes):
    img_shape = tf.cast(tf.shape(img), dtype=tf.float32)
    org_height, org_width = img_shape[0], img_shape[1]
    img, gt_boxes = randomly_apply_operation(expand_image, img, gt_boxes, org_height, org_width)
    min_overlap = get_raondom_min_overlap()

    begin, size, new_boundaries = tf.image.sample_distorted_bounding_box(
        tf.shape(img),
        bounding_boxes=tf.expand_dims(gt_boxes, 0),
        aspect_ratio_range=[0.5, 2.0],
        min_object_covered=min_overlap)

    img = tf.slice(img, begin, size)
    img = tf.image.resize(img, (org_height, org_width))
    gt_boxes = bbox_utils.renormalize_bboxes_with_min_max(gt_boxes, new_boundaries[0,0])
    
    return img, gt_boxes