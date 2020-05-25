# Step 1: Convert GTSRB ppm images to jpg
import os
from PIL import Image
import wget
#Get GTSRB data
wget.download('https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/FullIJCNN2013.zip')
unzip_command = 'unzip FullIJCNN2013.zip'
os.system(unzip_command)

image_data_dir = 'FullIJCNN2013'
ppm_images_list = [os.path.join(image_data_dir, i) for i in os.listdir(image_data_dir) if i.endswith('ppm')]

for ppm_image in ppm_images_list:
    im = Image.open(ppm_image)
    im.save(os.path.splitext(ppm_image)[0] + '.jpg')

# Create a tensorflow_object_detection_dataset folder
tfod_path = ('tensorflow_object_detection')
os.makedirs(tfod_path, exist_ok=True)
tfod_data_path = os.path.join(tfod_path, 'data')
os.makedirs(tfod_data_path, exist_ok=True)
# Create Label_map
label_map = '''
item {
  id: 1
  name: 'sign'
}
'''

label_map_path = os.path.join(tfod_data_path, 'labelmap.pbtxt')
with open(label_map_path, 'w') as f:
    f.write(label_map)

# %%

# Convert_ground_truth_text_to_dict.
with open(os.path.join(image_data_dir, 'gt.txt'), 'r') as f:
    lines = [i.strip() for i in f.readlines()]

gt_dict = {}
for line in lines:
    filename_ppm, xmin, ymin, xmax, ymax, _ = line.split(';')
    filename_jpg = os.path.splitext(filename_ppm)[0] + '.jpg'
    dict_for_file = gt_dict.get(filename_jpg, {'xmins': [],
                                               'ymins': [],
                                               'xmaxs': [],
                                               'ymaxs': [],
                                               'class_id': [],
                                               'class_text': []})
    dict_for_file['xmins'].append(int(xmin))
    dict_for_file['ymins'].append(int(ymin))
    dict_for_file['xmaxs'].append(int(xmax))
    dict_for_file['ymaxs'].append(int(ymax))
    dict_for_file['class_id'].append(1)
    dict_for_file['class_text'].append(b'sign')
    gt_dict[filename_jpg] = dict_for_file

# %%

# Read all images and convert it into tfrecords to train the object detection model.
import tensorflow as tf
import numpy as np
from object_detection.utils import dataset_util
import cv2


def create_tf_example(imagepath):
    im = np.array(Image.open(imagepath))
    height = im.shape[0]  # Image height
    width = im.shape[1]  # Image width
    filename = os.path.split(imagepath)[1]
    encoded_image_data = cv2.imencode('.jpg', im)[1].tostring()  # Encoded image bytes
    image_format = b'jpeg'

    xmins = [i / (width + 0.0) for i in
             gt_dict[filename]['xmins']]  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [i / (width + 0.0) for i in
             gt_dict[filename]['xmaxs']]  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [i / (height + 0.0) for i in
             gt_dict[filename]['ymins']]  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [i / (height + 0.0) for i in
             gt_dict[filename]['ymaxs']]  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = gt_dict[filename]['class_text']  # List of string class name of bounding box (1 per box)
    classes = gt_dict[filename]['class_id']  # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode()),
        'image/source_id': dataset_util.bytes_feature(filename.encode()),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


validation_split = 0.1
train_tfrecord_path = os.path.join(tfod_data_path, 'train.tfrecord')
val_tfrecord_path = os.path.join(tfod_data_path, 'val.tfrecord')

train_writer = tf.io.TFRecordWriter(train_tfrecord_path)
val_writer = tf.io.TFRecordWriter(val_tfrecord_path)

for filename in gt_dict.keys():
    tf_example = create_tf_example(os.path.join(image_data_dir, filename))
    if (np.random.rand() < 0.1):
        val_writer.write(tf_example.SerializeToString())
    else:
        train_writer.write(tf_example.SerializeToString())

train_writer.close()
val_writer.close()

wget.download('http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz','tensorflow_object_detection/models/ssd_mobilenet_v2_coco.tar.gz')
unzip_command = 'tar -xzf tensorflow_object_detection/models/ssd_mobilenet_v2_coco.tar.gz'
os.system(unzip_command)
mv_command = 'mv tensorflow_object_detection/models/ssd_mobilenet_v2_coco_2018_03_29 tensorflow_object_detection/models/ssd_mobilenet_v2_coco'
os.system(mv_command)

