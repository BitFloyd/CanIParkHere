import tensorflow as tf
import numpy as np
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
import train_traffic_sign_detector.config as config

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_path):
    f = tf.io.gfile.GFile(model_path, 'rb')
    graph_def = tf.compat.v1.GraphDef()
    # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    sess.graph.as_default()
    tf.compat.v1.import_graph_def(graph_def)
    # layers = [op.name for op in sess.graph.get_operations()]
    # print (layers)
    return sess


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # Run inference
    output_dict = {}
    image_tensor = model.graph.get_tensor_by_name('import/image_tensor:0')
    detection_boxes = model.graph.get_tensor_by_name('import/detection_boxes:0')
    detection_scores = model.graph.get_tensor_by_name('import/detection_scores:0')
    detection_classes = model.graph.get_tensor_by_name('import/detection_classes:0')
    num_detections = model.graph.get_tensor_by_name('import/num_detections:0')

    boxes, scores, classes, num_detections = model.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: [image]})
    output_dict['num_detections'] = num_detections
    output_dict['detection_scores'] = scores
    output_dict['detection_classes'] = classes
    output_dict['detection_boxes'] = boxes

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.

    num_detections = int(output_dict.pop('num_detections'))
    if (config.TF2):
        output_dict = {key: value[0, :num_detections].numpy() for key, value in output_dict.items()}
    else:
        output_dict = {key: value[0, :num_detections] for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    return output_dict

def visualize_detections_on_image(model, image_np):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        config.CATEGORY_INDEX,
        min_score_thresh=config.SCORE_THRESHOLD,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)

    return image_np


def create_candidate_boxes_in_frame(detection_dict, image_shape, score_threshold):
    candidate_boxes = []
    for idx, detection_score in enumerate(detection_dict['detection_scores']):
        if (detection_score >= score_threshold and detection_dict['detection_classes'][idx] in config.ID_LIST):
            box = detection_dict['detection_boxes'][idx]
            ymin, xmin, ymax, xmax = box
            ymin = int(ymin * image_shape[0])
            xmin = int(xmin * image_shape[1])
            ymax = int(ymax * image_shape[0])
            xmax = int(xmax * image_shape[1])
            candidate_boxes.append({'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax,
                                    'box_area': (xmax - xmin) * (ymax - ymin),
                                    'score': detection_score})
    return candidate_boxes