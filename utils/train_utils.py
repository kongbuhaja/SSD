import tensorflow as tf
import math
from utils import bbox_utils

SSD = {
    "vgg16": {
        "img_size": 300,
        "feature_map_shapes": [38, 19, 10, 5, 3, 1],
        "aspect_ratios": [[1., 2., 1./2.],
                          [1., 2., 1./2., 3., 1./3.],
                          [1., 2., 1./2., 3., 1./3.],
                          [1., 2., 1./2., 3., 1./3.],
                          [1., 2., 1./2.],
                          [1., 2., 1./2.]]
    },
    "mobilenet_v2": {
        "img_size": 300,
        "feature_map_shapes": [19, 10, 5, 3, 2, 1],
        # "img_size": 512,
        # "feature_map_shapes": [32, 16, 8, 4, 2, 1],
        "aspect_ratios": [[1., 2., 1./2.],
                          [1., 2., 1./2., 3., 1./3.],
                          [1., 2., 1./2., 3., 1./3.],
                          [1., 2., 1./2., 3., 1./3.],
                          [1., 2., 1./2.],
                          [1., 2., 1./2.]]
    }
}

def get_hyper_params(backbone, **kwargs):
    hyper_params=SSD[backbone]
    hyper_params["iou_threshold"] = 0.5
    hyper_params["neg_pos_ratio"] = 3
    hyper_params["loc_loss_alpha"] = 1
    hyper_params["variances"] = [0.1, 0.1, 0.2, 0.2]
    for key, value in kwargs.items():
        if key in hyper_params and value:
            hyper_params[key] = value

    return hyper_params

def scheduler(epoch):
    if epoch < 100:
        return 1e-3
    elif epoch < 125:
        return 1e-4
    elif epoch < 150:
        return 1e-5


def get_step_size(total_items, batch_size):
    return math.ceil(total_items / batch_size)

def generator(dataset, default_boxes, hyper_params):
    while True:
        for image_data in dataset:
            img, gt_boxes, gt_labels = image_data
            actual_deltas, actual_labels = calculate_actual_outputs(default_boxes, gt_boxes, gt_labels, hyper_params)
            yield img, (actual_deltas, actual_labels)

def calculate_actual_outputs(default_boxes, gt_boxes, gt_labels, hyper_params):
    # input
    #     prior_boxes=(8732,4) [y1,x1,y2,x2]
    #     gt_boxes=(n,g,4)
    #     gt_labels=(n,g)
    #     hyper_params=hyper_params
    total_labels = hyper_params["total_labels"]
    iou_threshold = hyper_params["iou_threshold"]
    variances = hyper_params["variances"]
    # (n,b,g)
    iou_map = bbox_utils.generate_iou_map(default_boxes, gt_boxes)
    # (n,b)
    merged_iou_map = tf.reduce_max(iou_map, axis=-1)
    # (n,b)
    max_indices_each_gt_box = tf.argmax(iou_map, axis=-1, output_type=tf.int32)
    # (n,b)
    pos_cond = tf.greater(merged_iou_map, iou_threshold)
    # (n,b,4)
    gt_boxes_map = tf.gather(gt_boxes, max_indices_each_gt_box, batch_dims=1)
    # (n,b,4) [0,0,0,0] if pos_cond=False else [y1,x1,y2,x2]
    expanded_gt_boxes = tf.where(tf.expand_dims(pos_cond, -1), gt_boxes_map, tf.zeros_like(gt_boxes_map))
    # (8732,4) [delta_y, delta_x, delta_h, delta_w]
    bbox_deltas = bbox_utils.get_deltas_from_bboxes(default_boxes, expanded_gt_boxes) / variances

    # (n,g)
    gt_labels_map = tf.gather(gt_labels, max_indices_each_gt_box, batch_dims=1)
    # (n,g) 0 if pos_cond=False else label
    expanded_gt_labels = tf.where(pos_cond, gt_labels_map, tf.zeros_like(gt_labels_map))
    # (n,g) -> (n,g,c)
    bbox_labels = tf.one_hot(expanded_gt_labels, total_labels)

    return bbox_deltas, bbox_labels