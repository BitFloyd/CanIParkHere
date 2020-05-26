import os
import tensorflow as tf
# patch tf1 into `utils.ops`
from object_detection.utils import ops as utils_ops
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

from object_detection.utils import label_map_util

MODEL_DIR_LOCAL = '/Users/sebyjacob/My_Projects/CanIParkHere/detector_model/exported_graph'
MODEL_PATH = os.path.join(MODEL_DIR_LOCAL, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(MODEL_DIR_LOCAL, 'labelmap.pbtxt')


def get_ids_of_interest(category_index, interests):
    id_list = []
    for key, value in category_index.items():
        # each value in category_index is like: {'id': 1, 'name': 'hand'}
        if (value['name'] in interests):
            id_list.append(value['id'])
    return id_list

SCORE_THRESHOLD=0.5
CATEGORY_INDEX = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
ID_LIST = get_ids_of_interest(CATEGORY_INDEX, interests=['hand'])
