import tensorflow as tf

def non_max_suppression(pred_bboxes, pred_labels, **kwargs):
    return tf.image.combined_non_max_suppression(pred_bboxes,
                                                 pred_labels,
                                                 **kwargs)

def generate_iou_map(bboxes, gt_boxes, transpose_perm=[0, 2, 1]):
    # input
    #     if 3d
    #     bboxes=(n,b,4)
    #     gtboxes=(n,g,4)
    # output
    #     intersection_arr / union_area=(n,b,g)

    gt_rank = tf.rank(gt_boxes)
    # if 3d gt_rank=3
    gt_expand_axis = gt_rank - 2

    # (n,b,1)
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(bboxes, 4, axis=-1)
    # (n,g,1)
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1)

    # (n,b)
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1)
    # (n,g)
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis=-1)
    
    # bbox_x1=(n,b,1) gt_x1=(n,g,1) x_top=(n,b,g)
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, transpose_perm))
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, transpose_perm))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, transpose_perm))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, transpose_perm))

    # (n,b,g)
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    # (n,b,1) + (n,1,g) = (n,b,g)
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, gt_expand_axis) - intersection_area)

    return intersection_area / union_area

def get_bboxes_from_deltas(default_boxes, deltas):
    all_pbox_width = default_boxes[...,  3] - default_boxes[..., 1]
    all_pbox_height = default_boxes[..., 2] - default_boxes[..., 0]
    all_pbox_ctr_x = default_boxes[..., 1] + 0.5 * all_pbox_width
    all_pbox_ctr_y = default_boxes[..., 0] + 0.5 * all_pbox_height

    all_bbox_width = tf.exp(deltas[..., 3]) * all_pbox_width
    all_bbox_height = tf.exp(deltas[..., 2]) * all_pbox_height
    all_bbox_ctr_x = (deltas[..., 1] * all_pbox_width) + all_pbox_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_pbox_height) + all_pbox_ctr_y

    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_ctr_y + (0.5 * all_bbox_height)
    x2 = all_bbox_ctr_x + (0.5 * all_bbox_width)

    return tf.stack([y1, x1, y2, x2], axis=-1)

def get_deltas_from_bboxes(bboxes, gt_boxes):
    # input
    #     bboxes=(b,4)  b=8732
    #     gt_boxes=(n,b,4)
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height

    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height

    bbox_width = tf.where(tf.equal(bbox_width, 0), 1e-3, bbox_width)
    bbox_height = tf.where(tf.equal(bbox_height, 0), 1e-3, bbox_height)
    # (n,b)
    delta_x = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.truediv((gt_ctr_x - bbox_ctr_x), bbox_width))
    delta_y = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.truediv((gt_ctr_y - bbox_ctr_y), bbox_height))
    delta_w = tf.where(tf.equal(gt_width, 0), tf.zeros_like(gt_width), tf.math.log(gt_width / bbox_width))
    delta_h = tf.where(tf.equal(gt_height, 0), tf.zeros_like(gt_height), tf.math.log(gt_height / bbox_height))
    # (n,b,4)
    return tf.stack([delta_y, delta_x, delta_h, delta_w], axis=-1)

def get_scale_for_nth_feature_map(k, m=6, scale_min=0.2, scale_max=0.9):
    return 0.1 if k==1 else scale_min + ((scale_max - scale_min) / (m - 2)) * (k - 2)

def generate_base_default_boxes(aspect_ratios, feature_map_index, total_feature_map):
    current_scale = get_scale_for_nth_feature_map(feature_map_index, m=total_feature_map)
    next_scale = get_scale_for_nth_feature_map(feature_map_index + 1, m=total_feature_map)
    base_default_boxes = []
    for aspect_ratio in aspect_ratios:
        height = current_scale / tf.sqrt(aspect_ratio)
        width = current_scale * tf.sqrt(aspect_ratio)
        base_default_boxes.append([-height/2, -width/2, height/2, width/2])

    height = width = tf.sqrt(current_scale * next_scale)
    base_default_boxes.append([-height/2, -width/2, height/2, width/2])
    return tf.cast(base_default_boxes, dtype=tf.float32)

def generate_default_boxes(feature_map_shapes, aspect_ratios):
    prior_boxes = []
    for i, feature_map_shape in enumerate(feature_map_shapes):
        # shape (n+1, 4) n=len of aspect_ratio
        base_default_boxes = generate_base_default_boxes(aspect_ratios[i], i+1, len(feature_map_shapes))

        strides = 1 / feature_map_shape
        grid_coords = tf.cast(tf.range(0, feature_map_shape) / feature_map_shape + strides / 2, dtype=tf.float32)
        grid_x, grid_y = tf.meshgrid(grid_coords, grid_coords)
        flat_grid_x, flat_grid_y = tf.reshape(grid_x, (-1,)), tf.reshape(grid_y, (-1,))

        grid_map = tf.stack([flat_grid_y, flat_grid_x, flat_grid_y, flat_grid_x], -1)

        # shape of prior_boxes_for_feature_map=f*f*(n+1), n=len of aspect_ratio
        prior_boxes_for_feature_map = tf.reshape(base_default_boxes, (1, -1, 4)) + tf.reshape(grid_map, (-1, 1, 4))
        prior_boxes_for_feature_map = tf.reshape(prior_boxes_for_feature_map, (-1, 4))

        prior_boxes.append(prior_boxes_for_feature_map)
    prior_boxes = tf.concat(prior_boxes, axis=0)
    # (8732,4)
    return tf.clip_by_value(prior_boxes, 0, 1)

def renormalize_bboxes_with_min_max(bboxes, min_max):
    y_min, x_min, y_max, x_max = tf.split(min_max, 4)
    renormalized_bboxes = bboxes - tf.concat([y_min, x_min, y_min, x_min], -1)
    renormalized_bboxes /= tf.concat([y_max-y_min, x_max-x_min, y_max-y_min, x_max-x_min], -1)
    return tf.clip_by_value(renormalized_bboxes, 0, 1)

def denormalize_bboxes(bboxes, height, width):
    y1 = bboxes[..., 0] * height
    x1 = bboxes[..., 1] * width
    y2 = bboxes[..., 2] * height
    x2 = bboxes[..., 3] * width
    return tf.round(tf.stack([y1, x1, y2, x2], axis=-1))