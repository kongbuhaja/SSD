import os
import argparse
import tensorflow as tf
from datetime import datetime

def get_log_path(model_type, custom_postfix=""):
    return "logs/{}{}/{}".format(model_type, custom_postfix, datetime.now().strftime("%Y%m%d-%H%M%S"))

def get_model_path(model_type):
    main_path = "trained"
    if not os.path.exists(main_path):
        os.makedirs(main_path)
    model_path = os.path.join(main_path, "ssd_{}_model_weights.h5".format(model_type))
    return model_path

def handle_args():
    parser = argparse.ArgumentParser(description="SSD: Single Shot MultiBox Detector Implementation")
    parser.add_argument("-handle-gpu", action="store_true", help="Tensorflow 2 GPU compatibility flag")
    parser.add_argument("--backbone", required=False,
                         default="vgg16",
                         metavar="['mobilenet_v2', 'vgg16']",
                         help="Which backbone used for the ssd")
    parser.add_argument("--custom_dataset_dir",
                         default=False,
                         help="custom_dataset dir")
    args = parser.parse_args()
    return args

def is_valid_backbone(backbone):
    assert backbone in ["mobilenet_v2", "vgg16"]

def handle_gpu_compatibility():
    try:
        gpus =tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(e)